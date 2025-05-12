# Genesis: A Compiler Framework for Hamiltonian Simulation on Hybrid CV-DV Quantum Computers

Genesis is a compiler framework for Hamiltonian simulation targeting hybrid continuous-variable (CV) and discrete-variable (DV) quantum systems. It supports multi-level compilation, Hybrid CV-DV domain-specific language (DSL), and hardware circuit mapping and routing. 

![Flowchart](doc/flowchart.png)

## 🔍 Compilation Pipeline

1. **Hamiltonian Parsing**: Translates a Hamiltonian from mathematical form into a DSL-based representation.
2. **Intermediate Representation (IR)**: Converts the DSL into an IR consisting of Pauli strings and operator expressions.
3. **Pattern Matching and Gate Synthesis**: Matches fermionic and bosonic operator terms and synthesizes them into **logical** CV-DV circuits in `CVDVQASM` format.
4. **Physical Mapping**: Maps logical circuits and Pauli terms to hardware-compliant physical circuits, and outputs the final(**physical**) `CVDVQASM` program(s).

## 🔥 News

## 🚀 Installation

### Prerequisites

* Python ≥ 3.8
* Java (for [ANTLR](https://www.antlr.org/))

We recommend using a Python virtual environment:

```shell
python3 -m venv Genesis
source Genesis/bin/activate
```

Then install dependencies:

```shell
pip install -r requirements.txt
```

Download ANTLR 4.13.0 into the `antlr` directory:

```shell
mkdir antlr
cd antlr
curl -O https://www.antlr.org/download/antlr-4.13.0-complete.jar
```

## 🔧 Building the Compiler

To generate the Python parser from the grammar file, run:

```shell
mkdir grammar generated
cd grammar
java -jar ../antlr/antlr-4.13.0-complete.jar -Dlanguage=Python3 -visitor hamiltonianDSL.g4 -o ../generated
```

The generated files will be saved to the `generated/` directory.

## 🧪 Benchmarks

Benchmark Hamiltonians (`*.ham`) are provided in the `benchmark/` directory. To define new Hamiltonians using Genesis DSL, see [Hamiltonian DSL Specification](doc/grammar.md).

## 📦 Usage

Genesis supports:

* **Single-file mode**
* **Batch JSON mode**
* Optional modes: `--debug`, `--stats`, and `--clean`

### Single File

```shell
python3 -m src.main benchmark/electronicVibration_small.ham
```

### Batch Mode

```shell
python3 -m src.main --batch-json tests/small_batch_jobs.json
```

Example `small_batch_jobs.json`:

```json
[
  {
    "input_file": "benchmark/hubbardHolstein_small.ham",
    "output_file": "output/hubbardHolstein_small.cvdvqasm"
  },
  {
    "input_file": "benchmark/electronicVibration_small.ham",
    "output_file": "output/electronicVibration_small.cvdvqasm"
  }
]
```

### Optional Flags

* `--debug`: Adds debug comments in output `.cvdvqasm` files
* `--stats`: Generates a JSON report of compilation statistics
* `--clean`: Removes intermediate files

## 🧩 Intermediate Tools

### Operator Pattern Matching

Transforms polynomial fermion/boson operators to hybrid CV-DV gate sequences, given an `.afe` file and an index to start ancilla qubits at. When called via command line, the output is printed out to standard output, which can then be piped elsewhere.

For example, using `tests/single_testcase/testcase1.afe`, which represents $e^{-i\frac{\pi}{2}b^\dagger b}$:

```c
exp(prod((-1.5708j),dagger(b(0)),b(0)))
```

We can run the operator pattern matching tool on this single testcase:

```shell
python3 -m src.pattern_match tests/single_testcase/testcase1.afe 0 > output/output.log
```

The output will be saved in `output/output.log`, `0` is the index of the potential ancilla qubit.

### Pattern Matching and Mapping test cases

Smaller test cases involving pattern matching and mapping can be ran by the commands below:
```shell
python3 -m tests.single_testcase.test_single_testcase > tests/single_testcase/test_single_testcase_outlog.txt
```

```shell
python3 -m tests.pauli_map_test.test_pauli_map > tests/pauli_map_test/test_pauli_map_outlog.txt
```

### CVDVQASM Mapping for Qubit Hamiltonians

To run the mapping algorithm on the given Qubit hamiltonians in `qubit_hamiltonians` folder, execute `src.decomp_molecule` akin to how one would run `main`. For example:

```shell
python3 -m src.decomp_molecule qubit_hamiltonians/LiH/BK_LiH_sto3g_4_electrons_12_spin_orbitals_Hamiltonian_631_paulis.txt
```

```shell
python3 -m src.decomp_molecule --batch-json tests/molecule_small_batch_jobs.json
```

The batch JSON shares the same format as the other example, but requires a different type of input file, looking like this:

```json
[
    {
        "input_file": "qubit_hamiltonians/LiH/BK_LiH_sto3g_4_electrons_12_spin_orbitals_Hamiltonian_631_paulis.txt",
        "output_file": "output/small_molecule_benchmark/BK_LiH_sto3g_4_electrons_12_spin_orbitals_Hamiltonian_631_paulis.cvdvqasm"
    },
    {
        "input_file": "qubit_hamiltonians/LiH/JW_LiH_sto3g_4_electrons_12_spin_orbitals_Hamiltonian_631_paulis.txt",
        "output_file": "output/small_molecule_benchmark/BK_LiH_sto3g_4_electrons_12_spin_orbitals_Hamiltonian_631_paulis.cvdvqasm"
    }
]
```

### 📝 Demo

A comprehensive set of demonstration usage examples and general benchmark evaluation is provided in the Demo Jupyter notebook [Demo](tests/demo.ipynb).

## 📬 Contact

* Eddy Z. Zhang — eddyzhengzhang \[at] gmail\.com
* Zihan Chen — zihan.chen.cs \[at] rutgers\.edu
* Henry Chen — hc867 \[at] scarletmail\.rutgers\.edu

## 📖 Citation

> Zihan Chen, Jiakang Li, Minghao Guo, Henry Chen, Zirui Li, Joel Bierman, Yipeng Huang, Huiyang Zhou, Yuan Liu, and Eddy Z. Zhang.
> *Genesis: A Compiler Framework for Hamiltonian Simulation on Hybrid CV-DV Quantum Computers*.
> In *Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA ’25)*, June 21–25, 2025, Tokyo, Japan.
