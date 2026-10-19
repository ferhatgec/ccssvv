# ccssvv

## RFC4180-compliant multidialect CSV parser module in C++23.

### Features:

* Written as a single-file C++23 module.
* RFC4180 compliance
* Dialect support
* AVX2 acceleration with SWAR fallback
* Zero-copy parsing and memory-mapped I/O when possible
* Exception-free error handling.

### Example: Parsing In-Memory Strings:

```c++
import ccssvv;
import std;

auto main() -> int {
  constexpr std::string_view input =
    "id,name,note\r\n"
    "1,Ada,\"likes \"\"engines\"\", and poetry\"\r\n"
    "2,Grace,\"multi\r\nline\"\r\n"
    "3,Linus,\r\n"
    "4,Gordon,loves physics\r\n"
    "5,Alyx,engineer\r\n";

  for(auto const& record : ccssvv::parse(input, ccssvv::Dialect::rfc4180())) {
    if(!record) {
      std::println("error: {}", record.error().message());
      return 1;
    }
    for(auto const field : record->fields()) {
      std::print("[{}]", field.view());
    }
    std::println("");
  }
}
```

### Example: Memory Mapped File Reading

```c++
import ccssvv;
import std;

auto read_file(std::filesystem::path const& path) -> void {
  auto reader = ccssvv::open_reader(path);
  if(!reader) {
    std::println("error: {}", reader.error().message());
    return;
  }

  for(auto const& record : *reader) {
    if(!record) {
      std::println("error: {}", record.error().message());
      return;
    }
    for(auto const field : record->fields()) {
      std::print("[{}]", field.view());
    }
    std::println("");
  }
}
```

### Example: Table/Column Projection

```c++
import ccssvv;
import std;

auto sum_prices(std::filesystem::path const& path) -> void {
  auto table = ccssvv::open_table(path, ccssvv::Dialect::rfc4180());
  if(!table) {
    std::println("error: {}", table.error().message());
    return;
  }

  auto price = table->column("price");
  if(!price) {
    std::println("error: {}", price.error().message());
    return;
  }

  auto total {0.0};
  for(auto const& cell : *price) {
    if(!cell) {
      std::println("error: {}", cell.error().message());
      return;
    }
    auto const text = cell->view();
    auto value {0.0};
    std::from_chars(text.data(), text.data() + text.size(), value);
    total += value;
  }
  std::println("total = {}", total);
}
```

### Example: Custom Dialects

```c++
import ccssvv;

ccssvv::Dialect tsv_dialect{
  .delimiter = '\t',
  .quote = '"',
  .newline = ccssvv::Newline::LfOrCrlf,
  .strict_text = false,
  .uniform_width = true,
  .skip_empty_lines = true,
  .skip_bom = true,
};

auto reader = ccssvv::open_reader("data.tsv", tsv_dialect);
```

### ccssvv is licensed under the terms of the MIT License.