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

auto read_columns() -> void {
  auto table = csv::Table<csv::ViewSource>::open(
    csv::ViewSource{sample}, csv::Dialect::rfc4180()
  );
  if(!table) {
    std::println("error: {}", table.error().message());
    return;
  }
  auto names = table->column("name");
  if(!names) {
    std::println("error: {}", names.error().message());
    return;
  }
  if(auto collected = names->collect()) {
    for(auto const& name : *collected) {
      std::println("name = {}", name);
    }
  }
}

auto sum_file_column(std::filesystem::path const& path) -> void {
  auto table = csv::open_table(path, csv::Dialect::rfc4180());
  if(!table) {
    std::println("error: {}", table.error().message());
    return;
  }
  auto price = table->column("price");
  if(!price) {
    std::println("error: {}", price.error().message());
    return;
  }
  auto total = 0.0;
  for(auto const& cell : *price) {
    if(!cell) {
      std::println("error: {}", cell.error().message());
      return;
    }
    auto const text = cell->view();
    auto value = 0.0;
    std::from_chars(text.data(), text.data() + text.size(), value);
    total += value;
  }
  std::println("total = {}", total);
}

} // namespace

auto main(int argc, char** argv) -> int {
  walk_records();
  read_columns();
  if(argc > 1) {
    sum_file_column(argv[1]);
  }
  return 0;
}