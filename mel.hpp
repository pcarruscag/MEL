// Copyright 2021-2024, Pedro Gomes
//
// This file is part of MEL.
//
// MEL is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MEL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with MEL.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>
#include <set>
#include <sstream>
#include <vector>
#include <limits>
#include <ctgmath>

#include "definitions.hpp"

namespace mel {
namespace internal {

/// Returns <true> if parentheses are balanced. Examples:
/// (((a+b)*c)) -> True
/// a)*2*(c -> False
template<class StringType>
bool BalancedParentheses(const StringType& expr) {
  int nest_level = 0;
  for (const auto c : expr) {
    if (c == '(') {
      nest_level += 1;
    } else if (c == ')') {
      nest_level -= 1;
    }
    // if the nest level becomes negative we know there is an imbalance
    if (nest_level < 0) {
      return false;
    }
  }
  return nest_level == 0;
}

/// Converts a string to a number of the desired type,
/// returns <true> if the conversion is successful.
template<class StringType, class NumberType>
bool ToNumber(const StringType& s, NumberType& n) {
  std::stringstream ss;
  ss << s;
  return BalancedParentheses(s) && static_cast<bool>(ss >> n);
}

/// Returns <true> if a string is convertible to number.
template<class StringType>
bool IsNumber(const StringType& s) {
  double n{};
  return ToNumber(s, n);
}

/// Replace all occurrences of "target" in a string.
template<class StringType>
void ReplaceAll(const StringType& target, const StringType& repl,
                StringType& str) {
  typename StringType::size_type pos = 0;
  while ((pos = str.find(target)) != StringType::npos) {
    str.replace(pos, target.size(), repl);
  }
}

/// Remove redundant parentheses around an expression. Examples:
/// (((a+b)*c)) -> (a+b)*c
template<class StringType>
StringType RemoveParentheses(StringType expr) {
  if (expr.size() > 2 && BalancedParentheses(expr)) {
    auto new_expr = StringType(expr.begin()+1, expr.end()-1);
    while ((expr.front() == '(') && BalancedParentheses(new_expr)) {
      expr = new_expr;
      if (expr.size() > 2) {
        new_expr = StringType(expr.begin()+1, expr.end()-1);
      }
    }
  }
  return expr;
}

/// Splits an expression into (op, rhs, lhs), where "op" is a character
/// in op_list. The split is not done if parentheses are not balanced,
/// or if the character preceeding "op" is in excl_chars. Examples:
/// "a+b" -> ("+", "a", "b")
/// "(a+b)*(a-b)" -> ("*", "(a+b)", "(a-b)")
template<class StringType>
std::array<StringType, 3> SplitAtOperation(const StringType& op_list,
    const StringType& excl_chars, const StringType& expr) {
  std::array<StringType, 3> ret{};
  if (expr.empty()) return ret;

  for (auto it = expr.end() - 1; it != expr.begin(); --it) {
    const auto rhs = StringType(expr.begin(), it+1);
    if (!BalancedParentheses(rhs)) {
      continue;
    }
    if (op_list.find(*it) != StringType::npos) {
      if (excl_chars.find(*(it-1)) == StringType::npos) {
        ret[0] = *it;
        ret[1] = StringType(expr.begin(), it);
        ret[2] = StringType(it+1, expr.end());
        return ret;
      }
    }
  }
  return ret;
}

/// Detects if a function from "func_list" is being applied to an expression.
/// Functions are of the form f() or f(,) (for binary functions).
/// The result is returned as (func, arg1, [arg2]). Examples:
/// "sqrt(pow(x,2)+1)" -> ("sqrt", "pow(x,2)+1")
template<class StringListType, class IntListType, class StringType>
std::array<StringType, 3> DetectFunction(const StringListType& func_list,
    const IntListType& narg_list, const StringType& expr) {
  std::array<StringType, 3> ret{};
  int i_func = -1;
  for (const auto& f : func_list) {
    const auto narg = narg_list[++i_func];
    if (expr.size() <= f.size() + 1) {
      continue;
    }
    // remove "f(" and ")"
    const auto it = expr.begin() + f.size() + 1;
    const auto start_expr = StringType(expr.begin(), it);
    const auto inner_expr = StringType(it, expr.end()-1);
    if (f+'(' == start_expr && BalancedParentheses(inner_expr)) {
      // consider "," an operation to detect binary functions
      const auto args = SplitAtOperation(StringType(","),
                                         StringType(), inner_expr);
      if (inner_expr.empty()) return ret;
      ret[0] = f;
      if (args[0].empty() && narg == 1) {
        ret[1] = inner_expr;
      } else if (!args[0].empty() && narg == 2) {
        ret[1] = args[1];
        ret[2] = args[2];
      } else {
        // Function used with the wrong number of arguments.
        ret[0] = expr;
      }
      return ret;
    }
  }
  return ret;
}

/// Modifies an expression to ensure that single character unary operations,
/// such as "-", look like a function, for example "-a" -> "-(a)". However,
/// literals are not modified, i.e. "-2" -> "-2". This also restores the
/// sign of scientific notation exponents.
template<class StringType>
StringType UnaryOpToUnaryFunc(const StringType& op_list,
                              const StringType& expr,
                              bool& is_number) {
  // recover the sign for scientific notation
  auto expr2 = expr;
  typename StringType::size_type pos;
  if ((pos = expr2.find('}')) != StringType::npos) expr2[pos] = '+';
  if ((pos = expr2.find('{')) != StringType::npos) expr2[pos] = '-';

  if (!IsNumber(expr2)) {
    is_number = false;
    if (op_list.find(expr.front()) != StringType::npos && expr[1] != '(') {
      return StringType({expr.front(), '('}) + StringType(expr.begin()+1,
             expr.end()) + ')';
    } else {
      return expr;
    }
  }
  is_number = true;
  return expr2;
}

/// Replaces the sign of scientific number exponents by }(+) or {(-) to avoid
/// splitting numbers when detecting operations.
template<class StringType>
void MarkScientificNotation(StringType& expr) {
  for (auto it = expr.begin(); it != expr.end(); ++it) {
    // Find where a number might start.
    if (possible_num_starts.find(*it) == str_t::npos) continue;
    // Advance to the next character,
    if (++it == expr.end()) break;
    // that can be a unary operation,
    if (unary_ops.find(*it) != str_t::npos) ++it;
    // or a sequence of valid digits,
    while (it != expr.end() && valid_digits.find(*it) != str_t::npos) ++it;
    if (it == expr.end()) break;
    // followed by "e" or "E",
    if (*it != 'e' && *it != 'E') continue;
    if (++it == expr.end()) break;
    // and then the sign we need to replace.
    if (*it == '+') *it = '}';
    if (*it == '-') *it = '{';
  }
}

/// Applies a series of substitution rules to an expression to make it
/// compatible with the parsing rules.
template<class StringListType, class StringType>
void Preprocess(const StringListType& rules, const StringListType& subs,
                StringType& expr) {
  auto old = StringType();
  while (old != expr) {
    old = expr;
    auto subs_it = std::begin(subs);
    for (const auto& rule : rules) {
      ReplaceAll(rule, *subs_it, expr);
      ++subs_it;
    }
  }
}

/// Finds string symbols, within quotation marks and possibly with spaces.
template <class StringType>
std::set<StringType> FindStrings(const StringType& expr) {
  std::set<StringType> strings;
  for (auto i = expr.find('"', 0); i < expr.size();) {
    const auto j = expr.find('"', i + 1);
    if (j < expr.size()) {
      strings.emplace(expr.begin() + i, expr.begin() + j + 1);
      i = expr.find('"', j + 1);
    } else {
      break;
    }
  }
  return strings;
}

/// Finds an applicable rule to an expression by trying all in the right order.
template<class StringType>
std::array<StringType, 3> ApplyRules(StringType expr) {
  if (expr.size() > 1 && no_ops.find(expr.front()) != str_t::npos) {
    expr = StringType(expr.begin()+1, expr.end());
  }
  expr = RemoveParentheses(expr);

  auto result = SplitAtOperation(type_one_ops_comm, type_two_ops, expr);
  if (result[0].empty()) {
    result = SplitAtOperation(type_one_ops_non_comm, type_two_ops, expr);
  }
  if (result[0].empty()) {
    result = SplitAtOperation(type_two_ops_comm, StringType(), expr);
  }
  if (result[0].empty()) {
    result = SplitAtOperation(type_two_ops_non_comm, StringType(), expr);
  }
  // The operations above are binary, thus enforce that a RHS exists.
  if (!result[0].empty() && result[2].empty()) {
    result[1].clear();
    // 1+, 1*, etc. are convertible to numbers, '' are added to prevent that.
    result[0] = '\'' + expr + '\'';
  }
  if (result[0].empty()) {
    bool is_number = false;
    expr = UnaryOpToUnaryFunc(unary_ops, expr, is_number);
    if (!is_number) {
      result = DetectFunction(funcs, nargs, expr);
    }
  }
  if (result[0].empty()) {
    // symbol or number
    result[0] = expr;
  }
  return result;
}

/// Converts the text representation of the operation into an operation code.
template<class StringType>
OpCode StringToOpCode(const StringType& str) {
  auto b = std::begin(supported_operations);
  return static_cast<OpCode>(std::find(b, std::end(supported_operations),
                                       str) - b);
}

/// Builds an expression tree by recursively extracting operations and building
/// subtrees for their lhs and rhs expressions. The symbols are also extracted.
template<class StringType, class StringListType, class TreeType,
         class IntListType>
void BuildExpressionTree(const StringType& expr, StringListType& symbols,
                         TreeType& tree, IntListType& n_children) {
  auto& node = tree.nodes[tree.size];
  auto& n_child = n_children[tree.size];
  const auto result = internal::ApplyRules(expr);

  if (result[1].empty()) {
    typename TreeType::type value;
    if (ToNumber(result[0], value)) {
      node.type = OpCode::NUMBER;
      node.val = value;
      n_child = 0;
    } else {
      node.type = OpCode::SYMBOL;
      // Find the index of the symbol, or append it to the list.
      const auto pos = std::find(symbols.begin(), symbols.end(), result[0]);
      node.symbol_id = static_cast<int>(pos - symbols.begin());
      if (pos == symbols.end()) {
        symbols.push_back(result[0]);
      }
      n_child = 0;
    }
  } else {
    // The node is an expression with 1 or 2 children.
    node.type = StringToOpCode(result[0]);
    assert(node.type != OpCode::NOOP);

    node.child.left = ++tree.size;
    tree.nodes[node.child.left].level = static_cast<short>(node.level + 1);
    BuildExpressionTree(result[1], symbols, tree, n_children);
    n_child += n_children[node.child.left] + 1;

    if (!result[2].empty()) {
      node.child.right = ++tree.size;
      tree.nodes[node.child.right].level = static_cast<short>(node.level + 1);
      BuildExpressionTree(result[2], symbols, tree, n_children);
      n_child += n_children[node.child.left] + 1;
    } else {
      node.child.right = -1;
    }
  }
}

/// Prints the nodes of a tree to a stream.
template<class TreeType, class StringListType, class StreamType>
void PrintTreeNodes(const TreeType& tree, const StringListType& symbols,
                    StreamType& stream) {
  for (int i = 0; i < tree.size; ++i) {
    const auto& node = tree.nodes[i];
    switch (node.type) {
    case OpCode::NUMBER:
      stream << i << "  L" << node.index << "  " << node.val << '\n';
      break;
    case OpCode::SYMBOL:
      stream << i << "  L" << node.index << "  "
             << symbols[node.symbol_id] << '\n';
      break;
    case OpCode::NOOP:
      assert(false);
      break;
    default:
      const auto& op = supported_operations[static_cast<int>(node.type)];
      stream << i << "  L" << node.index << "  " << op << "  "
             << node.child.left << "  " << node.child.right << '\n';
    }
  }
}

/// Prints a representation of a tree to a stream.
template<class TreeType, class StringListType, class StreamType>
void PrintExpressionTree(const TreeType& tree, int i,
                         const StringListType& symbols,
                         int level, StreamType& stream) {
  for (int k = 0; k < level; ++k) {
    stream << "  ";
  }
  const auto& node = tree.nodes[i];

  switch (node.type) {
  case OpCode::NUMBER:
    stream << node.val << '\n';
    break;
  case OpCode::SYMBOL:
    stream << symbols[node.symbol_id] << '\n';
    break;
  case OpCode::NOOP:
    assert(false);
    break;
  default:
    const auto& op = supported_operations[static_cast<int>(node.type)];
    stream << op << '\n';
    PrintExpressionTree(tree, node.child.left, symbols, level+1, stream);
    if (node.child.right >= 0) {
      PrintExpressionTree(tree, node.child.right, symbols, level+1, stream);
    }
    break;
  }
}

template <OptimMode>
struct EvalStackSize {
  static constexpr int size = max_tree_size;
};

template <>
struct EvalStackSize<OptimMode::STACK_SIZE> {
  static constexpr int size = std::ceil(log2(max_tree_size));
};

/// Evaluates an expression tree.
template<class ReturnType, class TreeType, class FunctionType>
ReturnType EvaluateExpressionTree(const TreeType& tree,
                                  const FunctionType& index_to_value) {
  // Because a node only depends on higher-level nodes, we can start evaluating
  // them from the highest level (bottom of the tree) until we arrive at the
  // top (final result). Note that there are no dependencies within each level.
  // This avoids recursion and thus is faster, at the expense of using stack
  // space, potentially for all possible intermediate results.
  std::array<ReturnType, EvalStackSize<TreeType::mode>::size> v;

  for (int j = tree.size - 1; j >= 0; --j) {
    const auto& node = tree.nodes[j];
    int i = j, left{}, right{};
    if (TreeType::mode == OptimMode::STACK_SIZE) {
      i = node.index;
      left = node.child_stack.left;
      right = node.child_stack.right;
    } else if (node.type != OpCode::NUMBER &&
               node.type != OpCode::SYMBOL) {
      left = node.child.left;
      right = node.child.right;
    }
    switch (node.type) {
    case OpCode::NUMBER:
      v[i] = static_cast<ReturnType>(node.val);
      break;
    case OpCode::SYMBOL:
      v[i] = index_to_value(node.symbol_id);
      break;
    case OpCode::ADD:
      v[i] = v[left] + v[right];
      break;
    case OpCode::SUB:
      if (right >= 0) {
        v[i] = v[left] - v[right];
      } else {
        v[i] = -v[left];
      }
      break;
    case OpCode::MUL:
      v[i] = v[left] * v[right];
      break;
    case OpCode::DIV:
      v[i] = v[left] / v[right];
      break;
      MEL_FUNCTION_IMPLEMENTATIONS(v[left], v[right])
    case OpCode::NOOP:
      assert(false);
    }
  }
  return v[0];
}

/// Remove common symbols and numbers by converting the nodes to NOOP.
/// The size of the tree then needs to be adjusted after sorting these
/// nodes to the end.
template <class TreeType>
void RemoveDuplicates(TreeType& tree) {
  for (int i = 0; i < tree.size; ++i) {
    auto& node_i = tree.nodes[i];
    if (node_i.type != internal::OpCode::SYMBOL &&
        node_i.type != internal::OpCode::NUMBER) {
      continue;
    }
    for (int j = 0; j < tree.size; ++j) {
      // For each function node check if the children use a value
      // or symbol equivalent to node "i".
      auto& node_j = tree.nodes[j];
      switch (node_j.type) {
      case internal::OpCode::NUMBER:
      case internal::OpCode::SYMBOL:
      case internal::OpCode::NOOP:
        break;
      default: {
        auto check_child = [&](int& k) {
          if (k < 0 || k == i) return;
          auto& node_k = tree.nodes[k];
          if (node_k.type != node_i.type) return;
          // If same symbol or value.
          if ((node_i.type == internal::OpCode::SYMBOL &&
              node_k.symbol_id == node_i.symbol_id) ||
              (node_i.type == internal::OpCode::NUMBER &&
              node_k.val == node_i.val)) {
            // Point to i instead of k, and change the type and level of
            // k such that it will be sorted last.
            k = i;
            node_i.level = std::max(node_i.level, node_k.level);
            node_k.level = std::numeric_limits<short>::max();
            node_k.type = internal::OpCode::NOOP;
          }
        };
        check_child(node_j.child.left);
        check_child(node_j.child.right);
        }
      }
    }
  }
}

/// Computes the index of each node of the tree in a depth-first traversal.
template <class TreeType, class IntListType>
void DepthFirstIndex(const int root, const TreeType& tree,
                     const IntListType& n_children, int& idx,
                     IntListType& index) {
  index[root] = idx++;
  const auto& node = tree.nodes[root];
  switch (node.type) {
    case OpCode::NUMBER:
    case OpCode::SYMBOL:
    case OpCode::NOOP:
      break;
    default: {
      // Take the child node with fewer nodes under it first.
      auto child = node.child;
      if (child.right >= 0 &&
          n_children[child.right] < n_children[child.left]) {
        std::swap(child.right, child.left);
      }
      DepthFirstIndex(child.left, tree, n_children, idx, index);
      if (child.right >= 0) {
        DepthFirstIndex(child.right, tree, n_children, idx, index);
      }
    }
  }
}

/// When optimizing for the required size of the evaluation stack, map the
/// destination location for nodes and source location for their children on
/// that stack.
template <class IntListType, class TreeType>
void MapEvaluationStack(IntListType& stack, TreeType& tree) {
  // Determine the position of each node on the evaluation stack.
  // After sorting by DFI the rules for pushing and popping are:
  // - Push if the level is greater or equal than the top of the stack.
  // - Pop the top one or two entries if the level is lower.
  int pos = 0;
  for (int i = tree.size - 1; i >= 0; --i) {
    auto& node = tree.nodes[i];
    if (pos > 0 && node.level < tree.nodes[stack[pos - 1]].level) {
      // Pop.
      pos -= node.child.right >= 0 ? 2 : 1;
    }
    // Push.
    stack[pos] = i;
    node.index = static_cast<short>(pos);
    ++pos;
  }
  // Determine the locations of child nodes on the stack,
  // this avoids indirection during evaluation.
  for (int i = 0; i < tree.size; ++i) {
    auto& node = tree.nodes[i];
    switch (node.type) {
    case internal::OpCode::NUMBER:
    case internal::OpCode::SYMBOL:
    case internal::OpCode::NOOP:
      break;
    default:
      node.child_stack.left =
          static_cast<int8_t>(tree.nodes[node.child.left].index);
      node.child_stack.right = static_cast<int8_t>(-1);
      if (node.child.right >= 0) {
        node.child_stack.right =
            static_cast<int8_t>(tree.nodes[node.child.right].index);
      }
    }
  }
}

} // namespace internal

/// Type for an expression tree. The result of parsing expressions and used
/// to evaluate them.
template<class NumberType, OptimMode Mode = internal::default_optim_mode>
struct ExpressionTree {
  using type = NumberType;
  static constexpr OptimMode mode = Mode;

  struct Node {
    // Type of the node, either as a leaf or as an operation.
    internal::OpCode type = internal::OpCode::NOOP;

    template <class IntType>
    struct Children {
      IntType left, right;
    };

    // Level of the node in the tree. Then the location of child
    // nodes on the evaluation stack (for OptimMode::STACK_SIZE).
    union {
      short level = 0;
      Children<int8_t> child_stack;
    };

    // Index of the node in the tree. Then the location of the
    // node on the evaluation stack (for OptimMode::STACK_SIZE).
    short index = 0;

    // If the node is a leaf, it is either a number or a symbol,
    // otherwise is has one or two child nodes.
    union {
      NumberType val;
      int symbol_id;
      Children<int> child;
    };

    bool operator<(const Node& other) const {
      return level != other.level ? level < other.level :
             static_cast<int>(type) < static_cast<int>(other.type);
    }
  };

  std::array<Node, internal::max_tree_size> nodes;
  int size = 0;
};

/// Preprocess an expression, create an expression tree for it, and extract its
/// symbols in the process. NumberType is the type used for stored constants
/// (i.e. literals).
template<class NumberType, OptimMode Mode = internal::default_optim_mode,
         class StringType, class StringListType>
ExpressionTree<NumberType, Mode> Parse(StringType expr,
                                       StringListType& symbols) {
  StringListType orig_strings, repl_strings;
  int i_repl = 0;
  for (const auto& str : internal::FindStrings(expr)) {
    const auto h = std::hash<StringType>{}(str) % 8192 + i_repl * 8192;
    orig_strings.emplace_back(str);
    repl_strings.emplace_back("mel_" + std::to_string(h));
    ++i_repl;
  }
  internal::Preprocess(orig_strings, repl_strings, expr);
  internal::Preprocess(internal::prep_rules, internal::prep_subs, expr);
  internal::MarkScientificNotation(expr);

  ExpressionTree<NumberType, Mode> tree{};
  for (int i = 0; i < internal::max_tree_size; ++i) {
    tree.nodes[i].index = static_cast<short>(i);
  }
  std::array<int, internal::max_tree_size> n_children{}, dfi{};
  internal::BuildExpressionTree(expr, symbols, tree, n_children);
  tree.size++;
  if (Mode == OptimMode::STACK_SIZE) {
    int idx = 0;
    internal::DepthFirstIndex(0, tree, n_children, idx, dfi);
  }

  // Undo the string replacements.
  for (auto& symbol : symbols) {
    const auto it =
        std::find(repl_strings.begin(), repl_strings.end(), symbol);
    if (it != repl_strings.end()) {
      symbol = orig_strings[std::distance(repl_strings.begin(), it)];
    }
  }
  // Early return if we are not optimizing the evaluation.
  if (Mode == OptimMode::NONE) return tree;

  // Sort nodes by their level in the tree, nodes in level i can be evaluated
  // with the values at level i+1. This makes the evaluation faster and it
  // allows removing the eliminated nodes easily. When minimizing the
  // evaluation stack size, sort by depth-first index instead.
  if (Mode == OptimMode::TREE_SIZE) {
    internal::RemoveDuplicates(tree);
    std::sort(tree.nodes.begin(), tree.nodes.begin() + tree.size);
  } else {
    using Node = typename ExpressionTree<NumberType, Mode>::Node;
    std::sort(tree.nodes.begin(), tree.nodes.begin() + tree.size,
              [&dfi](const Node& a, const Node& b) {
                return dfi[a.index] < dfi[b.index];
              });
  }

  // Renumber children after sorting.
  std::array<int, internal::max_tree_size> perm;
  for (int i = 0; i < internal::max_tree_size; ++i) {
    perm[tree.nodes[i].index] = i;
  }
  auto new_size = tree.size;
  for (int i = 0; i < tree.size; ++i) {
    auto& node = tree.nodes[i];
    switch (node.type) {
    case internal::OpCode::NUMBER:
    case internal::OpCode::SYMBOL:
      break;
    case internal::OpCode::NOOP:
      --new_size;
      break;
    default:
      node.child.left = perm[node.child.left];
      if (node.child.right >= 0) {
        node.child.right = perm[node.child.right];
      }
    }
  }
  if (Mode == OptimMode::TREE_SIZE) {
    tree.size = new_size;
  } else {
    auto& stack = n_children;
    internal::MapEvaluationStack(stack, tree);
  }
  return tree;
}

/// Prints a representation of a tree to a stream.
template<class TreeType, class StringListType, class StreamType>
void Print(const TreeType& tree, const StringListType& symbols,
           StreamType& stream) {
  internal::PrintExpressionTree(tree, 0, symbols, 0, stream);
}

/// Prints the nodes of a tree to a stream.
template<class TreeType, class StringListType, class StreamType>
void PrintNodes(const TreeType& tree, const StringListType& symbols,
                StreamType& stream) {
  internal::PrintTreeNodes(tree, symbols, stream);
}

/// Evaluates an expression. The functor "index_to_value" should map the index
/// of each symbol (order in the list produced by Parse) to its value
/// (int -> ReturnType). The return type does not need to be the same as the
/// type of number for the constants in the tree.
template<class ReturnType, class TreeType, class FunctionType>
ReturnType Eval(const TreeType& tree, const FunctionType& index_to_value) {
  return internal::EvaluateExpressionTree<ReturnType>(tree, index_to_value);
}

/// Overload of Eval, where the functor "symbol_to_value" should map each
/// symbol to its value (StringType -> ReturnType).
template<class ReturnType, class TreeType, class StringListType,
         class FunctionType>
ReturnType Eval(const TreeType& tree, const StringListType& symbols,
                const FunctionType& symbol_to_value) {
  auto index_to_value = [&](int i) {
    return symbol_to_value(symbols[i]);
  };
  return internal::EvaluateExpressionTree<ReturnType>(tree, index_to_value);
}

/// Overload of Eval, evaluates a raw expression (string) assuming it does not
/// contain symbols (provided for convenience).
template<class ReturnType, OptimMode Mode = OptimMode::NONE, class StringType>
ReturnType Eval(const StringType& expr) {
  std::vector<str_t> s;
  auto f = [&](int) {
    assert(false && "Unexpected symbol");
    return ReturnType{};
  };
  return internal::EvaluateExpressionTree<ReturnType>(
           Parse<ReturnType, Mode>(str_t(expr), s), f);
}

} // namespace mel
