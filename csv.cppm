// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Ferhat Geçdoğan.
//
module;

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

[[nodiscard]] inline auto describe(ErrorCode code) noexcept
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