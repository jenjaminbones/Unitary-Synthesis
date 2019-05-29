## Unitary Synthesis

The aim of this project is to synthesise a 2^n x 2^n unitary matrix into single qubit and controlled not gates.
The `compiler.py` module achieves this by bringing together the 3 compilation stages:

1) Decomposing an arbitrary unitary matrix into two-level unitary matrices (`two_level.py`)
2) Decomposing a two-level unitary matrix into a fully controlled operation (`fully_controlled.py`)
3) Decomposing a fully controlled operation into single qubit gates and CNOT gates (`single_cnot.py`)

The whole process can be seen via calling the `compile_unitary` method in `compiler.py`.

`util.py` defines several useful functions, particularly with regard to binary string maniuplation.

`gates.py` provides a framework for a gate object.

`circuit.py` contains a simple class for a circuit (a list of gates).

`kronecker.py` explores some interesting functions related to kronecker decompositons.

---

Throughout, qubit indices range from 1 to n (inclusive), and matrix entries range from 0 to 2^n -1.