// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Ferhat Geçdoğan.
//
module;

#if defined(__unix__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define CCSSVV_HAS_MMAP 1
#elifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  define CCSSVV_HAS_MMAP 1
#endif

#ifdef __AVX2__
#  include <immintrin.h>
#endif

export module ccssvv;

import std;

export namespace ccssvv {
enum class ScannerType : std::uint8_t {
  SWAR,
  AVX2,
};

enum class ErrorCode : std::uint8_t {
  InvalidDialect,
  UnterminatedQuote,
  StrayQuote,
  InvalidCharacterAfterQuote,
  InvalidCharacter,
  InvalidLineEnding,
  FieldCountMismatch,
  RecordTooLarge,
  IoFailure,
  MissingHeader,
  UnknownColumn,
  ColumnOutOfRange,
};

[[nodiscard]] constexpr auto describe(ErrorCode code) noexcept
  -> std::string_view {
  switch(code) {
    case ErrorCode::InvalidDialect:
      return "delimiter and quote must differ and must not be CR or LF";
    case ErrorCode::UnterminatedQuote:
      return "quoted field is never closed";
    case ErrorCode::StrayQuote:
      return "quote inside an unquoted field";
    case ErrorCode::InvalidCharacterAfterQuote:
      return "unexpected character after closing quote";
    case ErrorCode::InvalidCharacter:
      return "character not allowed by RFC 4180 TEXTDATA";
    case ErrorCode::InvalidLineEnding:
      return "line ending not permitted by the dialect";
    case ErrorCode::FieldCountMismatch:
      return "record has a different number of fields than the first record";
    case ErrorCode::RecordTooLarge:
      return "record exceeds the configured maximum size";
    case ErrorCode::IoFailure:
      return "I/O failure";
    case ErrorCode::MissingHeader:
      return "input is empty, no header record";
    case ErrorCode::UnknownColumn:
      return "no column with that name";
    case ErrorCode::ColumnOutOfRange:
      return "column index out of range";
  }
  return "unknown error";
}

struct Error {
  ErrorCode code{};
  std::size_t record{};
  std::size_t offset{};
  std::size_t expected{};
  std::size_t actual{};

  [[nodiscard]] auto message() const -> std::string {
    if(code == ErrorCode::FieldCountMismatch) {
      return std::format(
        "{} (expected {}, found {}; record {}, byte offset {})",
        describe(code),
        expected,
        actual,
        record,
        offset
      );
    }
    return std::format(
      "{} (record {}, byte offset {})", describe(code), record, offset
    );
  }
};

enum class Newline : std::uint8_t {
  Crlf,
  LfOrCrlf,
  Any,
};

struct Dialect {
  char delimiter{','};
  char quote{'"'};
  Newline newline{Newline::LfOrCrlf};
  bool strict_text{false};
  bool uniform_width{true};
  bool skip_empty_lines{false};
  bool skip_bom{true};

  [[nodiscard]] constexpr auto valid() const noexcept -> bool {
    auto const reserved = [](char c) noexcept -> bool {
      return c == '\r' || c == '\n';
    };
    return delimiter != quote && !reserved(delimiter) && !reserved(quote);
  }

  [[nodiscard]] static constexpr auto rfc4180() noexcept -> Dialect {
    return {.newline = Newline::Crlf, .strict_text = true};
  }
};

inline constexpr std::size_t npos = std::string_view::npos;

// extension point for vectorised kernels. both functions return the index of
// the first match at or after "from" or "npos". they must tolerate
// from == text.size() and must never read outside "text".
template <typename S>
concept Scanner =
  std::constructible_from<S, Dialect const&> &&
  requires(S const& scanner, std::string_view text, std::size_t from) {
    {
      scanner.find_structural(text, from)
    } noexcept -> std::same_as<std::size_t>;
    { scanner.find_quote(text, from) } noexcept -> std::same_as<std::size_t>;
  };

} // namespace ccssvv

namespace ccssvv::detail {

inline constexpr std::uint64_t lane_ones = 0x0101010101010101ULL;
inline constexpr std::uint64_t lane_low7 = 0x7F7F7F7F7F7F7F7FULL;

[[nodiscard]] constexpr auto broadcast(char c) noexcept -> std::uint64_t {
  return lane_ones * static_cast<std::uint8_t>(c);
}

[[nodiscard]] constexpr auto zero_lanes(std::uint64_t v) noexcept
  -> std::uint64_t {
  return ~(((v & lane_low7) + lane_low7) | v | lane_low7);
}

[[nodiscard]] constexpr auto first_lane(std::uint64_t mask) noexcept
  -> std::size_t {
  if constexpr(std::endian::native == std::endian::little) {
    return static_cast<std::size_t>(std::countr_zero(mask)) / 8;
  } else {
    return static_cast<std::size_t>(std::countl_zero(mask)) / 8;
  }
}

// only meaningful for Dialect::strict_text.
[[nodiscard]] inline auto find_non_text(
  std::string_view text, bool quoted
) noexcept -> std::size_t {
  auto const* const data = text.data();
  auto const size = text.size();
  auto i{0UZ};

#ifdef __AVX2__
  if(i + 32 <= size) {
    auto const min_printable = _mm256_set1_epi8(0x20); // ascii 0x20 = space
    auto const del_char = _mm256_set1_epi8(0x7F);
    auto const cr_char = _mm256_set1_epi8('\r');
    auto const lf_char = _mm256_set1_epi8('\n');

    do {
      auto const block =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(data + i));
      // signed comparison: bytes < 0x20 (0x00..0x1F) and >= 0x80 (-128..-1)
      // are all strictly less than 0x20 (32) in two's complement int8_t.
      auto bad = _mm256_or_si256(
        _mm256_cmpgt_epi8(min_printable, block),
        _mm256_cmpeq_epi8(block, del_char)
      );
      if(quoted) {
        auto const allowed_crlf = _mm256_or_si256(
          _mm256_cmpeq_epi8(block, cr_char), _mm256_cmpeq_epi8(block, lf_char)
        );
        bad = _mm256_andnot_si256(allowed_crlf, bad);
      }
      if(
        auto const mask = static_cast<std::uint32_t>(_mm256_movemask_epi8(bad));
        mask != 0
      ) {
        return i + static_cast<std::size_t>(std::countr_zero(mask));
      }
      i += 32;
    } while(i + 32 <= size);
  }
#endif

  for(; i < size; ++i) {
    auto const c = data[i];
    auto const byte = static_cast<unsigned char>(c);
    auto const ok =
      (byte >= 0x20 && byte <= 0x7E) || (quoted && (c == '\r' || c == '\n'));
    if(!ok) {
      return i;
    }
  }
  return npos;
}

} // namespace ccssvv::detail

export namespace ccssvv {

// portable 8-bytes-per-step scanner.
class SwarScanner {
 public:
  constexpr explicit SwarScanner(Dialect const& dialect) noexcept
      : delimiter_{dialect.delimiter},
        quote_{dialect.quote},
        lanes_{
          detail::broadcast(dialect.delimiter),
          detail::broadcast(dialect.quote),
          detail::broadcast('\r'),
          detail::broadcast('\n'),
        } {}

  [[nodiscard]] auto find_structural(
    std::string_view text, std::size_t from
  ) const noexcept -> std::size_t {
    auto const* const data = text.data();
    auto const size = text.size();
    auto i = from;
    for(; i + sizeof(std::uint64_t) <= size; i += sizeof(std::uint64_t)) {
      std::uint64_t word{};
      std::memcpy(&word, data + i, sizeof word);
      auto const hits = detail::zero_lanes(word ^ lanes_[0]) |
                        detail::zero_lanes(word ^ lanes_[1]) |
                        detail::zero_lanes(word ^ lanes_[2]) |
                        detail::zero_lanes(word ^ lanes_[3]);
      if(hits != 0) {
        return i + detail::first_lane(hits);
      }
    }
    for(; i < size; ++i) {
      if(is_structural(data[i])) {
        return i;
      }
    }
    return npos;
  }

  [[nodiscard]] auto find_quote(
    std::string_view text, std::size_t from
  ) const noexcept -> std::size_t {
    return text.find(quote_, from);
  }

  [[nodiscard]] static constexpr auto scanner_type() noexcept -> ScannerType {
    return ScannerType::SWAR;
  }

 private:
  [[nodiscard]] constexpr auto is_structural(char c) const noexcept -> bool {
    return c == delimiter_ || c == quote_ || c == '\r' || c == '\n';
  }

  char delimiter_{};
  char quote_{};
  std::array<std::uint64_t, 4> lanes_{};
};

#ifdef __AVX2__

// 32 bytes per step.
// the sub-32-byte tail falls back to SwarScanner so it never reads past "text"
class Avx2Scanner {
 public:
  constexpr explicit Avx2Scanner(Dialect const& dialect) noexcept
      : delimiter_{dialect.delimiter}, quote_{dialect.quote}, tail_{dialect} {}

  [[nodiscard]] auto find_structural(
    std::string_view text, std::size_t from
  ) const noexcept -> std::size_t {
    auto const* const data = text.data();
    auto const size = text.size();
    auto i = from;
    if(i + 32 <= size) {
      auto const delimiter_vec = _mm256_set1_epi8(delimiter_);
      auto const quote_vec = _mm256_set1_epi8(quote_);
      auto const cr_vec = _mm256_set1_epi8('\r');
      auto const lf_vec = _mm256_set1_epi8('\n');
      do {
        auto const block =
          _mm256_loadu_si256(reinterpret_cast<__m256i const*>(data + i));
        auto const hits = _mm256_or_si256(
          _mm256_or_si256(
            _mm256_cmpeq_epi8(block, delimiter_vec),
            _mm256_cmpeq_epi8(block, quote_vec)
          ),
          _mm256_or_si256(
            _mm256_cmpeq_epi8(block, cr_vec), _mm256_cmpeq_epi8(block, lf_vec)
          )
        );
        auto const mask =
          static_cast<std::uint32_t>(_mm256_movemask_epi8(hits));
        if(mask != 0) {
          return i + static_cast<std::size_t>(std::countr_zero(mask));
        }
        i += 32;
      } while(i + 32 <= size);
    }
    return tail_.find_structural(text, i);
  }

  // libc's memchr is already vectorised and runtime-dispatched.
  [[nodiscard]] auto find_quote(
    std::string_view text, std::size_t from
  ) const noexcept -> std::size_t {
    return text.find(quote_, from);
  }

  [[nodiscard]] static constexpr auto scanner_type() noexcept -> ScannerType {
    return ScannerType::AVX2;
  }

 private:
  char delimiter_{};
  char quote_{};
  SwarScanner tail_;
};

using DefaultScanner = Avx2Scanner;

#else
using DefaultScanner = SwarScanner;
#endif

// a Source exposes the not-yet-consumed bytes ("window"). views taken from the
// window stay valid until the next call to "refill".
template <typename S>
concept Source = requires(S& source, std::size_t count) {
  { std::as_const(source).window() } noexcept -> std::same_as<std::string_view>;
  { source.consume(count) } noexcept -> std::same_as<void>;
  { source.refill() } -> std::same_as<std::expected<bool, ErrorCode>>;
};

class ViewSource {
 public:
  constexpr explicit ViewSource(std::string_view text) noexcept : text_{text} {}

  [[nodiscard]] constexpr auto window() const noexcept -> std::string_view {
    return text_;
  }
  constexpr auto consume(std::size_t count) noexcept -> void {
    text_.remove_prefix(count);
  }
  [[nodiscard]] constexpr auto refill() noexcept
    -> std::expected<bool, ErrorCode> {
    return false;
  }

 private:
  std::string_view text_;
};

struct StreamOptions {
  std::size_t chunk_size{1UZ << 20UZ};
  std::size_t max_record_size{1UZ << 26UZ};
};

// for inputs that cannot be mapped (pipes, sockets, std::cin). only the
// unfinished tail of the buffer is moved between refills, never the chunk.
template <typename Stream>
  requires std::derived_from<std::remove_reference_t<Stream>, std::istream>
class StreamSource {
 public:
  explicit StreamSource(Stream stream, StreamOptions options = {})
      : stream_(std::forward<Stream>(stream)),
        limit_{std::max(options.max_record_size, options.chunk_size)},
        capacity_{std::max<std::size_t>(options.chunk_size, 1)},
        buffer_{std::make_unique_for_overwrite<char[]>(capacity_)} {}

  [[nodiscard]] auto window() const noexcept -> std::string_view {
    return {buffer_.get() + head_, tail_ - head_};
  }

  auto consume(std::size_t count) noexcept -> void { head_ += count; }

  [[nodiscard]] auto refill() -> std::expected<bool, ErrorCode> {
    if(eof_) {
      return false;
    }
    if(head_ == tail_) {
      head_ = tail_ = 0;
    } else if(head_ != 0) {
      std::memmove(buffer_.get(), buffer_.get() + head_, tail_ - head_);
      tail_ -= head_;
      head_ = 0;
    }
    if(tail_ == capacity_) {
      if(capacity_ >= limit_) {
        return std::unexpected{ErrorCode::RecordTooLarge};
      }
      grow(std::min(capacity_ * 2, limit_));
    }
    stream_.read(
      buffer_.get() + tail_, static_cast<std::streamsize>(capacity_ - tail_)
    );
    auto const got = static_cast<std::size_t>(stream_.gcount());
    tail_ += got;
    if(stream_.bad() || (stream_.fail() && !stream_.eof())) {
      return std::unexpected{ErrorCode::IoFailure};
    }
    eof_ = stream_.eof();
    return got != 0;
  }

 private:
  auto grow(std::size_t new_capacity) -> void {
    auto bigger = std::make_unique_for_overwrite<char[]>(new_capacity);
    std::memcpy(bigger.get(), buffer_.get(), tail_);
    buffer_ = std::move(bigger);
    capacity_ = new_capacity;
  }

  Stream stream_;
  std::size_t limit_{};
  std::size_t capacity_{};
  std::unique_ptr<char[]> buffer_;
  std::size_t head_{};
  std::size_t tail_{};
  bool eof_{};
};

template <typename S>
StreamSource(S&&) -> StreamSource<S>;

template <typename S>
StreamSource(S&&, StreamOptions) -> StreamSource<S>;

using FileSource = StreamSource<std::ifstream>;

[[nodiscard]] inline auto open_file(
  std::filesystem::path const& path, StreamOptions options = {}
) -> std::expected<FileSource, ErrorCode> {
  std::ifstream file{path, std::ios::binary};
  if(!file) {
    return std::unexpected{ErrorCode::IoFailure};
  }
  return FileSource{std::move(file), options};
}

#ifdef CCSSVV_HAS_MMAP

class MmapSource {
 public:
  [[nodiscard]] static auto map(std::filesystem::path const& path)
    -> std::expected<MmapSource, ErrorCode> {
#  ifdef _WIN32
    HANDLE hFile = CreateFileW(
      path.c_str(),
      GENERIC_READ,
      FILE_SHARE_READ,
      NULL,
      OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL,
      NULL
    );
    if(hFile == INVALID_HANDLE_VALUE) {
      return std::unexpected{ErrorCode::IoFailure};
    }

    LARGE_INTEGER size;
    if(!GetFileSizeEx(hFile, &size)) {
      CloseHandle(hFile);
      return std::unexpected{ErrorCode::IoFailure};
    }
    if(size.QuadPart == 0) {
      CloseHandle(hFile);
      return MmapSource{};
    }

    HANDLE hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);
    if(!hMap) {
      return std::unexpected{ErrorCode::IoFailure};
    }

    void* base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    if(!base) {
      return std::unexpected{ErrorCode::IoFailure};
    }

    return MmapSource{base, static_cast<std::size_t>(size.QuadPart)};
#  else
    auto const fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if(fd < 0) {
      return std::unexpected{ErrorCode::IoFailure};
    }

    struct stat info{};
    if(::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
      ::close(fd);
      return std::unexpected{ErrorCode::IoFailure};
    }
    auto const length = static_cast<std::size_t>(info.st_size);
    if(length == 0) {
      ::close(fd);
      return MmapSource{};
    }
    void* const base = ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if(base == MAP_FAILED) {
      return std::unexpected{ErrorCode::IoFailure};
    }

    ::madvise(base, length, MADV_SEQUENTIAL);
    return MmapSource{base, length};
#  endif
  }

  MmapSource(MmapSource const&) = delete;
  auto operator=(MmapSource const&) -> MmapSource& = delete;

  MmapSource(MmapSource&& other) noexcept
      : base_{std::exchange(other.base_, nullptr)},
        length_{std::exchange(other.length_, 0)},
        text_{std::exchange(other.text_, {})} {}

  auto operator=(MmapSource&& other) noexcept -> MmapSource& {
    if(this != &other) {
      release();
      base_ = std::exchange(other.base_, nullptr);
      length_ = std::exchange(other.length_, 0);
      text_ = std::exchange(other.text_, {});
    }
    return *this;
  }

  ~MmapSource() { release(); }

  [[nodiscard]] auto window() const noexcept -> std::string_view {
    return text_;
  }
  auto consume(std::size_t count) noexcept -> void {
    text_.remove_prefix(count);
  }
  [[nodiscard]] auto refill() noexcept -> std::expected<bool, ErrorCode> {
    return false;
  }

 private:
  MmapSource() noexcept = default;

  MmapSource(void* base, std::size_t length) noexcept
      : base_{base},
        length_{length},
        text_{static_cast<char const*>(base), length} {}

  auto release() noexcept -> void {
    if(base_ != nullptr) {
#  ifdef _WIN32
      UnmapViewOfFile(base_);
#  else
      ::munmap(base_, length_);
#  endif
      base_ = nullptr;
    }
  }

  void* base_{};
  std::size_t length_{};
  std::string_view text_;
};

using DefaultFileSource = MmapSource;

#else
using DefaultFileSource = FileSource;
#endif

[[nodiscard]] inline auto open_source(std::filesystem::path const& path)
  -> std::expected<DefaultFileSource, ErrorCode> {
#ifdef CCSSVV_HAS_MMAP
  return MmapSource::map(path);
#else
  return open_file(path);
#endif
}

} // namespace ccssvv

export namespace ccssvv {

class Text {
 public:
  Text(std::string_view borrowed) noexcept : storage_{borrowed} {}
  explicit Text(std::string owned) noexcept : storage_{std::move(owned)} {}

  [[nodiscard]] auto view() const noexcept -> std::string_view {
    return std::visit(
      [](auto const& value) noexcept -> std::string_view { return value; },
      storage_
    );
  }

  [[nodiscard]] auto owns() const noexcept -> bool {
    return std::holds_alternative<std::string>(storage_);
  }

 private:
  std::variant<std::string_view, std::string> storage_;
};

class Field {
 public:
  constexpr Field() noexcept = default;
  constexpr Field(std::string_view raw, char quote) noexcept
      : raw_{raw}, quote_{quote} {}

  [[nodiscard]] constexpr auto raw() const noexcept -> std::string_view {
    return raw_;
  }

  [[nodiscard]] constexpr auto quoted() const noexcept -> bool {
    return !raw_.empty() && raw_.front() == quote_;
  }

  // content without the enclosing quotes; doubled quotes are left as-is.
  [[nodiscard]] constexpr auto view() const noexcept -> std::string_view {
    return quoted() ? raw_.substr(1, raw_.size() - 2) : raw_;
  }

  [[nodiscard]] constexpr auto needs_unescape() const noexcept -> bool {
    return quoted() && view().contains(quote_);
  }

  // borrows when possible, owns only when doubled quotes have to be collapsed.
  [[nodiscard]] auto text() const -> Text {
    if(!needs_unescape()) {
      return Text{view()};
    }
    return Text{to_string()};
  }

  // reuses "buffer"'s capacity; the result views either the field or "buffer".
  [[nodiscard]] auto unescape_into(std::string& buffer) const
    -> std::string_view {
    if(!needs_unescape()) {
      return view();
    }
    unescape(view(), quote_, buffer);
    return buffer;
  }

  [[nodiscard]] auto to_string() const -> std::string {
    if(!needs_unescape()) {
      return std::string{view()};
    }
    std::string owned;
    unescape(view(), quote_, owned);
    return owned;
  }

 private:
  static auto unescape(std::string_view content, char quote, std::string& out)
    -> void {
    out.clear();
    out.reserve(content.size());
    for(;;) {
      auto const at = content.find(quote);
      if(at == npos) {
        out.append(content);
        return;
      }
      out.append(content.substr(0, at + 1));
      content.remove_prefix(std::min(at + 2, content.size()));
    }
  }

  std::string_view raw_;
  char quote_{'"'};
};

// where every field of one record starts and ends. "delimiters[k]" is the
// offset (from the start of "record") of the delimiter that follows field k,
// so field i is located in O(1). the reader fills the table while it validates.
struct FieldLayout {
  std::string_view record;
  std::size_t const* delimiters{};
  std::size_t count{};
  char quote{'"'};

  [[nodiscard]] constexpr auto at(std::size_t i) const noexcept -> Field {
    auto const begin = i == 0 ? 0UZ : delimiters[i - 1] + 1;
    auto const end = i + 1 == count ? record.size() : delimiters[i];
    return {record.substr(begin, end - begin), quote};
  }
};

class FieldIterator {
 public:
  using value_type = Field;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::random_access_iterator_tag;
  using iterator_category = std::input_iterator_tag;

  constexpr FieldIterator() noexcept = default;
  constexpr FieldIterator(FieldLayout layout, std::size_t index) noexcept
      : layout_{layout}, index_{index} {}

  [[nodiscard]] constexpr auto operator*() const noexcept -> Field {
    return layout_.at(index_);
  }

  [[nodiscard]] constexpr auto operator[](difference_type n) const noexcept
    -> Field {
    return layout_.at(offset_by(n));
  }

  constexpr auto operator++() noexcept -> FieldIterator& {
    ++index_;
    return *this;
  }

  constexpr auto operator++(int) noexcept -> FieldIterator {
    auto previous = *this;
    ++index_;
    return previous;
  }

  constexpr auto operator--() noexcept -> FieldIterator& {
    --index_;
    return *this;
  }

  constexpr auto operator--(int) noexcept -> FieldIterator {
    auto previous = *this;
    --index_;
    return previous;
  }

  constexpr auto operator+=(difference_type n) noexcept -> FieldIterator& {
    index_ = offset_by(n);
    return *this;
  }

  constexpr auto operator-=(difference_type n) noexcept -> FieldIterator& {
    return *this += -n;
  }

  [[nodiscard]] friend constexpr auto operator+(
    FieldIterator it, difference_type n
  ) noexcept -> FieldIterator {
    return it += n;
  }

  [[nodiscard]] friend constexpr auto operator+(
    difference_type n, FieldIterator it
  ) noexcept -> FieldIterator {
    return it += n;
  }

  [[nodiscard]] friend constexpr auto operator-(
    FieldIterator it, difference_type n
  ) noexcept -> FieldIterator {
    return it -= n;
  }

  [[nodiscard]] friend constexpr auto operator-(
    FieldIterator const& a, FieldIterator const& b
  ) noexcept -> difference_type {
    return static_cast<difference_type>(a.index_) -
           static_cast<difference_type>(b.index_);
  }

  [[nodiscard]] friend constexpr auto operator==(
    FieldIterator const& a, FieldIterator const& b
  ) noexcept -> bool {
    return a.index_ == b.index_;
  }

  [[nodiscard]] friend constexpr auto operator<=>(
    FieldIterator const& a, FieldIterator const& b
  ) noexcept -> std::strong_ordering {
    return a.index_ <=> b.index_;
  }

 private:
  [[nodiscard]] constexpr auto offset_by(difference_type n) const noexcept
    -> std::size_t {
    return static_cast<std::size_t>(static_cast<difference_type>(index_) + n);
  }

  FieldLayout layout_{};
  std::size_t index_{};
};

class FieldRange : public std::ranges::view_interface<FieldRange> {
 public:
  constexpr FieldRange() noexcept = default;
  constexpr explicit FieldRange(FieldLayout layout) noexcept
      : layout_{layout} {}

  [[nodiscard]] constexpr auto begin() const noexcept -> FieldIterator {
    return {layout_, 0};
  }
  [[nodiscard]] constexpr auto end() const noexcept -> FieldIterator {
    return {layout_, layout_.count};
  }
  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return layout_.count;
  }

 private:
  FieldLayout layout_{};
};

static_assert(std::ranges::random_access_range<FieldRange>);
static_assert(std::ranges::sized_range<FieldRange>);
static_assert(std::ranges::view<FieldRange>);

} // namespace ccssvv

namespace ccssvv::detail {

enum class Status : std::uint8_t {
  Record,
  End,
  NeedMore,
  Error,
};

struct Step {
  Status status{Status::NeedMore};
  std::size_t length{};
  std::size_t consumed{};
  std::size_t fields{};
  ErrorCode code{};
  std::size_t where{};
};

[[nodiscard]] constexpr auto incomplete() noexcept -> Step {
  return {.status = Status::NeedMore};
}

[[nodiscard]] constexpr auto finished() noexcept -> Step {
  return {.status = Status::End};
}

[[nodiscard]] constexpr auto complete(
  std::size_t length, std::size_t consumed, std::size_t fields
) noexcept -> Step {
  return {
    .status = Status::Record,
    .length = length,
    .consumed = consumed,
    .fields = fields,
  };
}

[[nodiscard]] constexpr auto failure(ErrorCode code, std::size_t where) noexcept
  -> Step {
  return {.status = Status::Error, .code = code, .where = where};
}

} // namespace ccssvv::detail

export namespace ccssvv {

class Record {
 public:
  constexpr Record() noexcept = default;

  constexpr Record(
    FieldLayout layout, std::size_t index, std::size_t offset
  ) noexcept
      : layout_{layout}, index_{index}, offset_{offset} {}

  [[nodiscard]] constexpr auto raw() const noexcept -> std::string_view {
    return layout_.record;
  }
  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t {
    return layout_.count;
  }
  [[nodiscard]] constexpr auto index() const noexcept -> std::size_t {
    return index_;
  }
  [[nodiscard]] constexpr auto offset() const noexcept -> std::size_t {
    return offset_;
  }
  [[nodiscard]] constexpr auto layout() const noexcept -> FieldLayout const& {
    return layout_;
  }
  [[nodiscard]] constexpr auto fields() const noexcept -> FieldRange {
    return FieldRange{layout_};
  }

  [[nodiscard]] constexpr auto field(std::size_t i) const noexcept
    -> std::optional<Field> {
    if(i >= layout_.count) {
      return std::nullopt;
    }
    return layout_.at(i);
  }

 private:
  FieldLayout layout_{};
  std::size_t index_{};
  std::size_t offset_{};
};

// single-pass input range of records. the record's field table lives in the
// reader and is reused, so a Record (and its FieldRange) is valid until the
// reader advances; the table only grows to the widest record seen.
template <Source Src, Scanner Scan = DefaultScanner>
class Reader {
 public:
  using Result = std::expected<Record, Error>;

  class Iterator {
   public:
    using value_type = Result;
    using difference_type = std::ptrdiff_t;

    Iterator() = default;
    explicit Iterator(Reader& reader) noexcept : reader_{&reader} {}

    [[nodiscard]] auto operator*() const noexcept -> Result const& {
      return *reader_->current_;
    }

    auto operator++() -> Iterator& {
      reader_->advance();
      return *this;
    }

    auto operator++(int) -> void { reader_->advance(); }

    [[nodiscard]] friend auto operator==(
      Iterator const& it, std::default_sentinel_t
    ) noexcept -> bool {
      return !it.reader_->still_has_value();
    }

   private:
    Reader* reader_{};
  };

  explicit Reader(Src source, Dialect dialect = {})
      : source_(std::move(source)), dialect_{dialect}, scanner_(dialect) {}

  [[nodiscard]] auto begin() -> Iterator {
    if(!started_) {
      started_ = true;
      advance();
    }
    return Iterator{*this};
  }

  [[nodiscard]] auto end() const noexcept -> std::default_sentinel_t {
    return {};
  }

  // pull interface. field-count mismatches are reported but iteration may
  // continue; every other error ends the stream.
  [[nodiscard]] auto next() -> std::optional<Result> {
    if(halted_) {
      return std::nullopt;
    }
    if(!primed_) {
      if(auto const failure = prime()) {
        return report(*failure);
      }
    }
    for(;;) {
      auto const window = source_.window();
      auto const step = scan(window, drained_);

      if(step.status == detail::Status::NeedMore) {
        auto const more = source_.refill();
        if(!more) {
          return report(
            Error{.code = more.error(), .record = index_ + 1, .offset = offset_}
          );
        }
        drained_ = !*more;
        continue;
      }
      if(step.status == detail::Status::End) {
        halted_ = true;
        return std::nullopt;
      }
      if(step.status == detail::Status::Error) {
        return report(
          Error{
            .code = step.code,
            .record = index_ + 1,
            .offset = offset_ + step.where,
          }
        );
      }

      auto const text = window.substr(0, step.length);
      auto const start = offset_;
      offset_ += step.consumed;
      source_.consume(step.consumed);

      if(dialect_.skip_empty_lines && step.length == 0) {
        continue;
      }
      ++index_;
      if(dialect_.uniform_width) {
        if(width_ == 0) {
          width_ = step.fields;
        } else if(step.fields != width_) {
          return report(
            Error{
              .code = ErrorCode::FieldCountMismatch,
              .record = index_,
              .offset = start,
              .expected = width_,
              .actual = step.fields,
            }
          );
        }
      }
      return Result{Record{
        FieldLayout{text, delimiters_.data(), step.fields, dialect_.quote},
        index_,
        start}};
    }
  }

  // materialises every remaining record with owned, unescaped fields.
  [[nodiscard]] auto collect()
    -> std::expected<std::vector<std::vector<std::string>>, Error> {
    std::vector<std::vector<std::string>> rows;
    for(auto const& item : *this) {
      if(!item) {
        return std::unexpected{item.error()};
      }
      auto& row = rows.emplace_back();
      row.reserve(item->size());
      for(auto const field : item->fields()) {
        row.push_back(field.to_string());
      }
    }
    return rows;
  }

  [[nodiscard]] auto dialect() const noexcept -> Dialect const& {
    return dialect_;
  }

  [[nodiscard]] auto still_has_value() const noexcept -> bool {
    return current_.has_value();
  }

 private:
  auto advance() -> void { current_ = next(); }

  [[nodiscard]] auto report(Error const& error) -> Result {
    halted_ = error.code != ErrorCode::FieldCountMismatch;
    return std::unexpected{error};
  }

  [[nodiscard]] auto prime() -> std::optional<Error> {
    primed_ = true;
    if(!dialect_.valid()) {
      return Error{.code = ErrorCode::InvalidDialect};
    }
    if(!dialect_.skip_bom) {
      return std::nullopt;
    }
    constexpr std::string_view bom{"\xEF\xBB\xBF"};
    while(source_.window().size() < bom.size() && !drained_) {
      auto const more = source_.refill();
      if(!more) {
        return Error{.code = more.error(), .record = 1, .offset = 0};
      }
      drained_ = !*more;
    }
    if(source_.window().starts_with(bom)) {
      source_.consume(bom.size());
      offset_ += bom.size();
    }
    return std::nullopt;
  }

  // validates one record and records where each field ends, without
  // materialising any field. with "final_ == false" the window may end
  // mid-record: the answer is then "NeedMore" whenever the bytes seen so
  // far cannot decide the outcome.
  [[nodiscard]] auto scan(std::string_view text, bool final_) -> detail::Step {
    auto const size = text.size();
    if(size == 0) {
      return final_ ? detail::finished() : detail::incomplete();
    }

    auto const quote = dialect_.quote;
    auto const delimiter = dialect_.delimiter;
    auto const is_boundary = [&](char c) noexcept -> bool {
      return c == delimiter || c == '\r' || c == '\n';
    };

    delimiters_.clear();
    auto pos = 0UZ;

    for(;;) {
      if(pos < size && text[pos] == quote) {
        auto const content = pos + 1;
        pos = content;
        for(;;) {
          pos = scanner_.find_quote(text, pos);
          if(pos == npos) {
            return final_ ? detail::failure(
                              ErrorCode::UnterminatedQuote, content - 1
                            )
                          : detail::incomplete();
          }
          if(pos + 1 == size && !final_) {
            return detail::incomplete();
          }
          if(pos + 1 < size && text[pos + 1] == quote) {
            pos += 2;
            continue;
          }
          break;
        }
        if(dialect_.strict_text) {
          auto const bad =
            detail::find_non_text(text.substr(content, pos - content), true);
          if(bad != npos) {
            return detail::failure(ErrorCode::InvalidCharacter, content + bad);
          }
        }
        ++pos;
        if(pos < size && !is_boundary(text[pos])) {
          return detail::failure(ErrorCode::InvalidCharacterAfterQuote, pos);
        }
      } else {
        auto const begin = pos;
        pos = scanner_.find_structural(text, pos);
        auto const stop = pos == npos ? size : pos;
        if(dialect_.strict_text) {
          auto const bad =
            detail::find_non_text(text.substr(begin, stop - begin), false);
          if(bad != npos) {
            return detail::failure(ErrorCode::InvalidCharacter, begin + bad);
          }
        }
        if(pos == npos) {
          pos = size;
        } else if(text[pos] == quote) {
          return detail::failure(ErrorCode::StrayQuote, pos);
        }
      }

      auto const fields = delimiters_.size() + 1;
      if(pos == size) {
        return final_ ? detail::complete(size, size, fields)
                      : detail::incomplete();
      }
      auto const c = text[pos];
      if(c == delimiter) {
        delimiters_.push_back(pos);
        ++pos;
        continue;
      }
      if(c == '\r') {
        if(pos + 1 == size && !final_) {
          return detail::incomplete();
        }
        if(pos + 1 < size && text[pos + 1] == '\n') {
          return detail::complete(pos, pos + 2, fields);
        }
        if(dialect_.newline != Newline::Any) {
          return detail::failure(ErrorCode::InvalidLineEnding, pos);
        }
        return detail::complete(pos, pos + 1, fields);
      }
      if(dialect_.newline == Newline::Crlf) {
        return detail::failure(ErrorCode::InvalidLineEnding, pos);
      }
      return detail::complete(pos, pos + 1, fields);
    }
  }

  Src source_;
  Dialect dialect_;
  Scan scanner_;
  std::vector<std::size_t> delimiters_;
  std::optional<Result> current_;
  std::size_t offset_{};
  std::size_t index_{};
  std::size_t width_{};
  bool primed_{};
  bool started_{};
  bool drained_{};
  bool halted_{};
};

[[nodiscard]] inline auto parse(std::string_view text, Dialect dialect = {})
  -> Reader<ViewSource> {
  return Reader<ViewSource>{ViewSource{text}, dialect};
}

// memory-mapped where the platform supports it, buffered stream otherwise.
[[nodiscard]] inline auto open_reader(
  std::filesystem::path const& path, Dialect dialect = {}
) -> std::expected<Reader<DefaultFileSource>, Error> {
  auto source = open_source(path);
  if(!source) {
    return std::unexpected{Error{.code = source.error()}};
  }
  return Reader<DefaultFileSource>{std::move(*source), dialect};
}

// always streams through a bounded buffer, whatever the platform.
[[nodiscard]] inline auto open_stream_reader(
  std::filesystem::path const& path,
  Dialect dialect = {},
  StreamOptions options = {}
) -> std::expected<Reader<FileSource>, Error> {
  auto source = open_file(path, options);
  if(!source) {
    return std::unexpected{Error{.code = source.error()}};
  }
  return Reader<FileSource>{std::move(*source), dialect};
}

static_assert(Scanner<SwarScanner>);
static_assert(Scanner<DefaultScanner>);
static_assert(Source<ViewSource>);
static_assert(Source<StreamSource<std::istream&>>);
static_assert(Source<DefaultFileSource>);
static_assert(std::ranges::input_range<Reader<ViewSource>>);

} // namespace ccssvv

namespace ccssvv::detail {

struct ColumnPicker {
  std::size_t index{};

  [[nodiscard]] auto operator()(std::expected<Record, Error> const& item) const
    -> std::expected<Field, Error> {
    if(!item) {
      return std::unexpected{item.error()};
    }
    if(auto const field = item->field(index)) {
      return *field;
    }
    return std::unexpected{Error{
      .code = ErrorCode::ColumnOutOfRange,
      .record = item->index(),
      .offset = item->offset(),
    }};
  }
};

} // namespace ccssvv::detail

export namespace ccssvv {

// lazy single column over the rows that remain in a reader.
template <typename R>
class Column : public std::ranges::view_base {
  using View =
    std::ranges::transform_view<std::ranges::ref_view<R>, detail::ColumnPicker>;

 public:
  Column(R& reader, std::size_t index)
      : view_{std::ranges::ref_view<R>{reader}, detail::ColumnPicker{index}} {}

  [[nodiscard]] auto begin() -> std::ranges::iterator_t<View> {
    return view_.begin();
  }
  [[nodiscard]] auto end() -> std::ranges::sentinel_t<View> {
    return view_.end();
  }

  [[nodiscard]] auto collect()
    -> std::expected<std::vector<std::string>, Error> {
    return collect([](Field field) -> std::string {
      return field.to_string();
    });
  }

  template <std::invocable<Field> Projection>
  [[nodiscard]] auto collect(Projection projection) -> std::
    expected<std::vector<std::invoke_result_t<Projection, Field>>, Error> {
    std::vector<std::invoke_result_t<Projection, Field>> values;
    for(auto const& item : *this) {
      if(!item) {
        return std::unexpected{item.error()};
      }
      values.push_back(std::invoke(projection, *item));
    }
    return values;
  }

 private:
  View view_;
};

// reads only the header up front (one copy). rows stay in the source until
// they are iterated, projected through "column()", or collected.
template <Source Src, Scanner Scan = DefaultScanner>
class Table {
 public:
  using reader_type = Reader<Src, Scan>;

  [[nodiscard]] static auto open(Src source, Dialect dialect = {})
    -> std::expected<Table, Error> {
    reader_type reader{std::move(source), dialect};
    auto first = reader.next();
    if(!first) {
      return std::unexpected{Error{.code = ErrorCode::MissingHeader}};
    }
    if(!*first) {
      return std::unexpected{first->error()};
    }
    auto const& layout = (*first)->layout();
    std::string header{layout.record};
    std::vector<std::size_t> ends(
      layout.delimiters, layout.delimiters + (layout.count - 1)
    );
    auto const width = layout.count;
    return Table{std::move(reader), std::move(header), std::move(ends), width};
  }

  [[nodiscard]] auto headers() const noexcept -> FieldRange {
    return FieldRange{FieldLayout{
      header_,
      header_ends_.data(),
      width_,
      reader_.dialect().quote,
    }};
  }

  [[nodiscard]] auto width() const noexcept -> std::size_t { return width_; }

  [[nodiscard]] auto column_index(std::string_view name) const
    -> std::optional<std::size_t> {
    std::string scratch;
    auto index{0UZ};
    for(auto const field : headers()) {
      if(field.unescape_into(scratch) == name) {
        return index;
      }
      ++index;
    }
    return std::nullopt;
  }

  [[nodiscard]] auto rows() noexcept -> reader_type& { return reader_; }

  [[nodiscard]] auto column(std::size_t index)
    -> std::expected<Column<reader_type>, Error> {
    if(index >= width_) {
      return std::unexpected{Error{.code = ErrorCode::ColumnOutOfRange}};
    }
    return Column<reader_type>{reader_, index};
  }

  [[nodiscard]] auto column(std::string_view name)
    -> std::expected<Column<reader_type>, Error> {
    auto const index = column_index(name);
    if(!index) {
      return std::unexpected{Error{.code = ErrorCode::UnknownColumn}};
    }
    return column(*index);
  }

 private:
  Table(
    reader_type reader,
    std::string header,
    std::vector<std::size_t> header_ends,
    std::size_t width
  )
      : reader_{std::move(reader)},
        header_{std::move(header)},
        header_ends_{std::move(header_ends)},
        width_{width} {}

  reader_type reader_;
  std::string header_;
  std::vector<std::size_t> header_ends_;
  std::size_t width_;
};

using FileTable = Table<DefaultFileSource>;
using StreamTable = Table<FileSource>;

[[nodiscard]] inline auto open_table(
  std::filesystem::path const& path, Dialect dialect = {}
) -> std::expected<FileTable, Error> {
  auto source = open_source(path);
  if(!source) {
    return std::unexpected{Error{.code = source.error()}};
  }
  return FileTable::open(std::move(*source), dialect);
}

[[nodiscard]] inline auto open_stream_table(
  std::filesystem::path const& path,
  Dialect dialect = {},
  StreamOptions options = {}
) -> std::expected<StreamTable, Error> {
  auto source = open_file(path, options);
  if(!source) {
    return std::unexpected{Error{.code = source.error()}};
  }
  return StreamTable::open(std::move(*source), dialect);
}

} // namespace ccssvv