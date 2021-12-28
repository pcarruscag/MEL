# MEL

MEL (Math Expression Library) is a small (~500 loc) header-only C++11 library to parse strings into math expression objects that can be evaluated at runtime, by substituting symbols (e.g. `x`) by runtime values.
It can be used, for example, to implement user-defined functions (UDF) in a larger code, in a self-contained way.

![Unit Tests](https://github.com/pcarruscag/mel/actions/workflows/c-cpp.yml/badge.svg)

## Usage

The two main functions of the library are `Parse` and `Eval`.
 - **Parse**: Creates an expression object (a tree) for an input string and extracts the symbols into a list, a symbol is a sub-string that cannot be split into more math operations, or interpreted directly as a number. The type (`float`, `double`, etc.) for such constants is a template parameter of the function.
 - **Eval**: Evaluates an expression object, the mapping from symbol to value can be specified in two ways, either via a `string -> value` functor, or more efficiently via a `index -> value` functor. In the latter, "index" is the position of a symbol in the list of symbols created by `Parse`. The return type for `Eval` is a template parameter, and can be different from the one used in `Parse` (provided the latter can be converted into the former).

An additional function, `Print`, can be used to visualize the expression objects produced by `Parse`.

### Example

```c++
#include <cmath>

#include "mel.hpp"

const std::string expr = "a + b*x + c*pow(x,2)";
std::vector<std::string> symbols;
const auto tree = mel::Parse<double>(expr, symbols);

// Assign values to symbols, here we know the order is "a", "b", "x", "c", but
// in a real application this process of (efficiently) mapping symbol indices
// to indices of recognized symbols is more involved.
std::vector<double> values = {1, -1, 0, 0.5};
auto idx_to_val = [&values](int i) {
  return values[i];
};

// Evaluate for different values of x.
for (double x = 0; x <= 10; x += 0.1) {
  values[2] = x;
  std::cout << mel::Eval<double>(tree, idx_to_val) << '\n';
}
```

### Default math functions and customization

In the default configuration, MEL parses the four arithmetic operations (`+-*/`), power and trigonometric functions from [cmath](https://www.cplusplus.com/reference/cmath/), `log`, `exp`, `fabs`, `fmin`, and `fmax`.
If the type used for `Eval` only supports simple arithmetic, the macro `MEL_ONLY_ARITHMETIC_OPS` can be defined before including `mel.hpp`.
The macros `MEL_SUPPORTED_FUNCTIONS`, `MEL_FUNCTION_CODES`, and `MEL_FUNCTION_IMPLEMENTATIONS`, can be used to fully customize the supported math functions, see definitions.hpp.

## Performance

MEL expressions are between 1 and 100 times slower to evaluate than native C++.
The best-case scenario is for expressions that use a large number of functions (`pow`, `exp`, etc.) relative to the number of symbols, or when the expression is evaluated in a bandwidth-bound context.
The worst-case scenario is for expressions with many repeated symbols (which implies common sub expressions that are not optimized in MEL) used in a compute-bound context.

## License

[LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.html)

