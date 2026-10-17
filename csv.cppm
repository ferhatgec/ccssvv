// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Ferhat Geçdoğan.
//
module;

#ifdef __AVX2__
#  include <immintrin.h>
#endif

export module csv;

import std;

export namespace csv {

enum class ErrorCode : std::uint8_t {
  invalid_dialect,
  unterminated_quote,
  stray_quote,
  invalid_character_after_quote,
  invalid_character,
  invalid_line_ending,
  field_count_mismatch,
  record_too_large,
  io_failure,
  missing_header,
  unknown_column,
  column_out_of_range,
};

[[nodiscard]] constexpr auto describe(ErrorCode code) noexcept
  -> std::string_view {
  switch(code) {
    case ErrorCode::invalid_dialect:
      return "delimiter and quote must differ and must not be CR or LF";
    case ErrorCode::unterminated_quote:
      return "quoted field is never closed";
    case ErrorCode::stray_quote:
      return "quote inside an unquoted field";
    case ErrorCode::invalid_character_after_quote:
      return "unexpected character after closing quote";
    case ErrorCode::invalid_character:
      return "character not allowed by RFC 4180 TEXTDATA";
    case ErrorCode::invalid_line_ending:
      return "line ending not permitted by the dialect";
    case ErrorCode::field_count_mismatch:
      return "record has a different number of fields than the first record";
    case ErrorCode::record_too_large:
      return "record exceeds the configured maximum size";
    case ErrorCode::io_failure:
      return "I/O failure";
    case ErrorCode::missing_header:
      return "input is empty, no header record";
    case ErrorCode::unknown_column:
      return "no column with that name";
    case ErrorCode::column_out_of_range:
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
    if(code == ErrorCode::field_count_mismatch) {
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
  crlf,
  lf_or_crlf,
  any,
};

struct Dialect {
  char delimiter{','};
  char quote{'"'};
  Newline newline{Newline::lf_or_crlf};
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
    return {.newline = Newline::crlf, .strict_text = true};
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

static_assert(Scanner<SwarScanner>);
static_assert(Scanner<DefaultScanner>);
} // namespace csv