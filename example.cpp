import csv;
import std;

auto main() -> int {
  constexpr auto dialect = csv::Dialect::rfc4180();
  constexpr std::string_view sample =
    "0123456789abcdefghijklmnopqrstuvwxyz,quoted_field,\"hello \"\"world\"\"\"\r\n";

  csv::SwarScanner const swar{dialect};
  csv::DefaultScanner const fast{dialect};

  auto pos = std::size_t{0};
  while(pos <= sample.size()) {
    auto const s_hit = swar.find_structural(sample, pos);
    auto const f_hit = fast.find_structural(sample, pos);
    if(s_hit != f_hit) {
      std::println("mismatch at {}: swar={}, fast={}", pos, s_hit, f_hit);
      return 1;
    }
    if(s_hit == csv::npos) {
      break;
    }
    std::println(
      "structural char '{}' found at offset {}",
      sample[s_hit] == '\r' ? 'R' : (sample[s_hit] == '\n' ? 'N' : sample[s_hit]),
      s_hit
    );
    pos = s_hit + 1;
  }

  auto const q1 = fast.find_quote(sample, 0);
  std::println("first quote at offset {}", q1);
  return 0;
}