// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Ferhat Geçdoğan.
//
module;

#if defined(__unix__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define CSV_HAS_MMAP 1
#elifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  define CSV_HAS_MMAP 1
#endif

#ifdef __AVX2__
#  include <immintrin.h>
#endif

export module csv;

import std;

export namespace csv {

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

} // namespace csv

namespace csv::detail {

inline constexpr std::uint64_t lane_ones = 0x0101010101010101ULL;
inline constexpr std::uint64_t lane_low7 = 0x7F7F7F7F7F7F7F7FULL;

namespace {
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
[[nodiscard]] auto find_non_text(std::string_view text, bool quoted) noexcept
  -> std::size_t {
  auto const* const data = text.data();
  auto const size = text.size();
  auto i = std::size_t{0};

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
} // namespace
} // namespace csv::detail

export namespace csv {

// portable 8-bytes-per-step scanner.
class SwarScanner {
 public:
  constexpr explicit SwarScanner(Dialect const& dialect) noexcept
      : delimiter_{dialect.delimiter},
        quote_{dialect.quote},
        lanes_{detail::broadcast(dialect.delimiter),
               detail::broadcast(dialect.quote),
               detail::broadcast('\r'),
               detail::broadcast('\n')} {}

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
  explicit Avx2Scanner(Dialect const& dialect) noexcept
      : quote_{dialect.quote},
        delimiter_vec_{_mm256_set1_epi8(dialect.delimiter)},
        quote_vec_{_mm256_set1_epi8(dialect.quote)},
        cr_vec_{_mm256_set1_epi8('\r')},
        lf_vec_{_mm256_set1_epi8('\n')},
        tail_{dialect} {}

  [[nodiscard]] auto find_structural(
    std::string_view text, std::size_t from
  ) const noexcept -> std::size_t {
    auto const* const data = text.data();
    auto const size = text.size();
    auto i = from;
    if(i + 32 <= size) {
      do {
        auto const block =
          _mm256_loadu_si256(reinterpret_cast<__m256i const*>(data + i));
        auto const hits = _mm256_or_si256(
          _mm256_or_si256(
            _mm256_cmpeq_epi8(block, delimiter_vec_),
            _mm256_cmpeq_epi8(block, quote_vec_)
          ),
          _mm256_or_si256(
            _mm256_cmpeq_epi8(block, cr_vec_), _mm256_cmpeq_epi8(block, lf_vec_)
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

 private:
  char quote_{};
  __m256i delimiter_vec_{};
  __m256i quote_vec_{};
  __m256i cr_vec_{};
  __m256i lf_vec_{};
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
  std::size_t chunk_size{1U << 20};
  std::size_t max_record_size{1U << 26};
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

#ifdef CSV_HAS_MMAP

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
#ifdef CSV_HAS_MMAP
  return MmapSource::map(path);
#else
  return open_file(path);
#endif
}

static_assert(Scanner<SwarScanner>);
static_assert(Scanner<DefaultScanner>);
static_assert(Source<ViewSource>);
static_assert(Source<StreamSource<std::istream&>>);
static_assert(Source<DefaultFileSource>);
} // namespace csv