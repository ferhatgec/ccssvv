import csv;
import std;

auto main() -> int {
  constexpr auto dialect = csv::Dialect::rfc4180();
  constexpr std::string_view raw = "id,name,note\r\n1,Ada,math\r\n2,Grace,compiler\r\n";

  csv::ViewSource view_src{raw};
  std::println("view initial size: {}", view_src.window().size());
  view_src.consume(14);
  std::println("view after consuming header: {}", view_src.window());

  std::istringstream iss{std::string{raw}};
  csv::StreamSource stream_src{
    std::move(iss), csv::StreamOptions{.chunk_size = 16, .max_record_size = 64}
  };

  csv::DefaultScanner const scanner{dialect};
  std::size_t total_consumed{};

  for(;;) {
    auto const win = stream_src.window();
    if(win.empty()) {
      auto const refilled = stream_src.refill();
      if(!refilled) {
        std::println("stream error: {}", csv::describe(refilled.error()));
        return 1;
      }
      if(!*refilled) {
        break;
      }
      continue;
    }
    auto const hit = scanner.find_structural(win, 0);
    auto const step = (hit == csv::npos) ? win.size() : hit + 1;
    stream_src.consume(step);
    total_consumed += step;
  }

  std::println("stream total consumed bytes: {} (expected {})", total_consumed, raw.size());
  return 0;
}