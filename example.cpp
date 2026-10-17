import csv;
import std;

auto main() -> int {
  constexpr auto rfc = csv::Dialect::rfc4180();
  static_assert(rfc.valid());

  constexpr csv::Dialect invalid{.delimiter = '"', .quote = '"'};
  static_assert(!invalid.valid());

  constexpr csv::Error err{
    .code = csv::ErrorCode::field_count_mismatch,
    .record = 3,
    .offset = 42,
    .expected = 4,
    .actual = 2,
  };
  std::println("{}", err.message());
  return 0;
}