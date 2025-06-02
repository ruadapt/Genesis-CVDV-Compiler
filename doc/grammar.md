# Hamiltonian DSL (H-DSL) Specification

This document specifies the **Hamiltonian Domain-Specific Language (H-DSL)** implemented in the `hamiltonianDSL.g4` grammar. It enables compact representation of quantum Hamiltonians with fermionic and bosonic operators.

## 🔍 Example

```c
Const t = 1;
Const U = 1;
Const g = 1;

Range i = [0, 10, 1];
Range j = [0, 10, 1];
Range sigma = [0, 2, 1];

Result = - t * Sum_over(i, j, sigma){FC[i][sigma] * FA[j][sigma]}
         + U * Sum_over(i){BC[i] * BA[i]}
         + g * Sum_over(i, sigma){TensorProd(FN[i][sigma], BC[i] + BA[i])};
```

This corresponds to the Hamiltonian:

$$
H = -t \sum_{i,j,\sigma} c^\dagger_{i\sigma} c_{j\sigma} + U \sum_i b^\dagger_i b_i + g \sum_{i,\sigma} \hat{n}_{i\sigma} (b_i^\dagger + b_i)
$$

where `FC`/`FA` denote fermionic creation/annihilation, `BC`/`BA` for bosons, and `FN` for fermionic number operators. Use `TensorProd` to combine fermionic and bosonic terms.

---

## 📌 1. Constants and Ranges

### Constants

```c
Const <IDENTIFIER> = <int> | <float>;
Const U = 1.5707963;
```

### Ranges

```c
Range <IDENTIFIER> = [ start , stop , step ];
Range i = [0, L, 1]; // i = 0 to L-1
```

* `stop` is exclusive, like Python's `range()`.

---

## 🧮 2. Arithmetic and Tensor Expressions

Operator precedence (lowest to highest):

| Level | Operators           | Associativity |
| ----- | ------------------- | ------------- |
| 1     | `+`, `-`            | left          |
| 2     | `*`, `/`            | left          |
| 3     | `^` (power)         | right         |
| 4     | Unary `-`           | left          |
| 5     | Grouped expressions | -             |

* `TensorProd(op1, ..., opN)` computes a Kronecker product.
* `imag` denotes the imaginary unit $i$.
* Complex constants like `Const z = 2 + 3*imag;` are allowed.
* Currently, the main program doesn't support any complex `Const` or `Range` values.

---

## 🧠 3. Quantum Operators

| Symbol     | Meaning                         |
| ---------- | ------------------------------- |
| `FC`, `FA` | Fermionic creation/annihilation |
| `FN`       | Fermionic number operator       |
| `BC`, `BA` | Bosonic creation/annihilation   |
| `Pauli_*`  | Single-qubit Pauli operators    |

Example:

```c
FC[i][1];
TensorProd(FC[i][1], FA[j][0]);
Pauli_Z[j][sigma];
```

`FN[i+1]`, `FC[i+L]` are valid as indices support expressions.

### Spin Index `sigma`

**Note:** The fermionic spin index must be named `sigma`.

We currently use the special index `sigma` to identify the 0-1 spin variable for fermionic operators.

---

## ∑ 4. Accumulation Expressions

| Syntax                        | Meaning                    |
| ----------------------------- | -------------------------- |
| `Sum_over(vars){expr}`        | Summation                  |
| `Prod_over(vars){expr}`       | Product                    |
| `TensorProd_over(vars){expr}` | Tensor product (Kronecker) |

Example:

```c
Sum_over(i, j){ op[i] * op[j] }
TensorProd_over(i){ op[i] }
```

* `Sum_over(...){...}` does **not** support nesting.
* `TensorProd_over` can be nested within `Sum_over`.

### Redundant Permutations

Redundant permutations are skipped when loop indices coincide, which means, if multiple ranges variables are the same(except `sigma`), the redundant permutations are skipped.

```c
op[i][sigma] * op[j][sigma] // will be skipped if i == j
```

---

## 🔚 5. Statements and Semicolons

Each top-level assignment must end in `;`:

```c
Result = ... ;
```

We use the assignment operator `=` to assign the result to identifier, and interpret the identifier as the result of the compilation.

---

## 🧾 6. Complete Example

This builds a hybrid Hamiltonian with fermion–fermion, boson–boson, and fermion–boson terms, we will use the Hubbard-Holstein model as an example:

$$
H = -t \sum_{\langle i,j, \sigma \rangle } c_{i,\sigma}^\dagger c_{j,\sigma} + U \sum_i \hat{n}_{i, \uparrow} \hat{n}_{i, \downarrow} + \sum_{i} b_i^\dagger b_i + g \sum_{\langle i, \sigma \rangle} \hat{n}_{i, \sigma} (b_i^\dagger + b_i)
$$

1. First, we define the constants and ranges.
2. Then, we assign the result as a sum of terms, use four `Sum_over` to accumulate the terms.
3. Each term is a product of operators,
    1. `-t * Sum_over(i, j, sigma){FC[i][sigma] * FA[j][sigma]}` represents two fermionic operators.
    2. `U * Sum_over(i){FN[i][1] * FN[i][0]}` represents two fermionic number operators.
    3. `Sum_over(i){BC[i] * BA[i]}` represents two bosonic operators.
    4. `g * Sum_over(i, sigma){TensorProd(FN[i][sigma], BC[i] + BA[i])}` represents a fermionic number operator tensor product with bosonic operators.
4. Then this `Result` identifier will be interpreted, and the compiler will generate the corresponding Hamiltonian circuit at last.

```c
Const t = 1.5707963;
Const U = 1.5707963;
Const g = 1.5707963;
Const L = 4;

Range i = [0, L, 1];
Range j = [0, L, 1];
Range sigma = [0, 2, 1];

Result = - t * Sum_over(i, j, sigma){FC[i][sigma] * FA[j][sigma]}
         + U * Sum_over(i){FN[i][1] * FN[i][0]}
         + Sum_over(i){BC[i] * BA[i]}
         + g * Sum_over(i, sigma){TensorProd(FN[i][sigma], BC[i] + BA[i])};
```

---

## 🧱 7. Lexical Summary

| Type         | Example                       | Token        |
| ------------ | ----------------------------- | ------------ |
| Identifier   | `U`, `i`, `sigma`             | `IDENTIFIER` |
| Number       | `3.14`, `1e-4`                | `NUMBER`     |
| Imaginary    | `imag`                        | `IMAG`       |
| Accumulation | `Sum_over` `Prod_over` `TensorProd_over` | keywords     |
| Quantum Ops  | `FC`, `FA`, `FN`, `BC`, `BA`, `Pauli_X`...         | quantum op   |
| Operators    | `+`, `-`, `*`, `/`, `^`, `=`  | operators    |
| Delimiters   | `()[]{},;`                    | symbols      |
| Comments     | `// ...` or `/* ... */`       | skipped      |

---

For detailed grammar rules, see the full [ANTLR grammar](../grammar/hamiltonianDSL.g4).

For questions or feedback, please contact the Genesis compiler team.
