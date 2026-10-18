import csv;
import std;

namespace {

constexpr std::string_view sample =
  "id,name,note\r\n"
  "1,Ada,\"likes \"\"engines\"\", and poetry\"\r\n"
  "2,Grace,\"multi\r\nline\"\r\n"
  "3,Linus,\r\n";

auto walk_records() -> void {
  std::string scratch;
  for(auto const& item : csv::parse(sample, csv::Dialect::rfc4180())) {
    if(!item) {
      std::println("error: {}", item.error().message());
      return;
    }
    for(auto const field : item->fields()) {
      std::print("[{}]", field.unescape_into(scratch));
    }
    std::println("");
  }
}

auto collect_all() -> void {
  auto reader = csv::parse(sample, csv::Dialect::rfc4180());
  auto const rows = reader.collect();
  if(!rows) {
    std::println("error: {}", rows.error().message());
    return;
  }
  std::println("collected {} records", rows->size());
}

} // namespace

auto main() -> int {
  walk_records();
  collect_all();
  return 0;
}