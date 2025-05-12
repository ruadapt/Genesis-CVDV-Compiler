from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

import numpy as np
import json


# Geometry: https://cccbdb.nist.gov/geom3x.asp?method=8&basis=20
h2o_atom = "O 0.0 0.0 0.1358; H 0.0 0.7706 -0.5432; H 0.0 -0.77-6 -0.5432"

# Geometry: https://cccbdb.nist.gov/geom3x.asp?method=8&basis=20
nh3_atom = "N 0.0 0.0 0.1446; H 0.0 0.9465 -0.3375; H 0.8197 -0.4733 -0.3375; H -0.8197 -0.4733 -0.3375"

# Geometry: https://cccbdb.nist.gov/geom3x.asp?method=8&basis=20
benzene_atom = "C 0.0 1.4093 0.0; C 1.2204 0.7046 0.0; C 1.2204 -0.7046 0.0; C 0.0 -1.4093 0.0; C -1.2204 -0.7046 0.0; C -1.2204 0.7046 0.0; H 0.0 2.5076 0.0; H 2.1716 1.2538 0.0; H 2.1716 -1.2538 0.0; H 0.0 -2.5076 0.0; H -2.1716 -1.2538 0.0; H -2.1716 1.2538 0.0"

# Geometry: https://cccbdb.nist.gov/geom3x.asp?method=8&basis=20
ethylene_atom = "C 0.0 0.0 0.6665; C 0.0 0.0 -0.6665; H 0.0 0.9301 1.2485; H 0.0 -0.9301 1.2485; H 0.0 -0.9301 -1.2485; H 0.0 0.9301 -1.2485"

# Geometry: https://cccbdb.nist.gov/geom3x.asp?method=8&basis=20
lih_atom = 'Li 0.0 0.0 0.0; H 0.0 0.0 1.526'

# Geometry: https://cccbdb.nist.gov/geom3x.asp?method=8&basis=20
beh2_atom = 'Be 0.0 0.0 0.0; H 0.0 0.0 1.3039; H 0.0 0.0 -1.3039'

# Geometry: https://cccbdb.nist.gov/geom3x.asp?method=8&basis=20
n2_atom = 'N 0.0 0.0 0.0; N 0.0 0.0 1.1807'

# Geometry: https://cccbdb.nist.gov/geom3x.asp?method=8&basis=20
c2_atom = 'C 0.0 0.0 0.0; C 0.0 0.0 1.2632'

def get_hamiltonians(molecule, basis, active_space_transformer = None):

    if molecule == 'H2O':
        
        atom = h2o_atom

    elif molecule == 'NH3':

        atom = nh3_atom

    elif molecule == 'ethylene':

        atom = ethylene_atom

    elif molecule == 'benzene':

        atom = benzene_atom

    elif molecule == "LiH":

        atom = lih_atom

    elif molecule == "BeH2":

        atom = beh2_atom

    elif molecule == "N2":

        atom = n2_atom

    elif molecule == "C2":

        atom = c2_atom

    driver = PySCFDriver(
        atom=atom,
        basis=basis,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    problem = driver.run()

    if active_space_transformer is not None:

        problem = active_space_transformer.transform(problem)

    fermionic_op = problem.hamiltonian.second_q_op()

    print(f'orbital occupancies: {problem.orbital_occupations}')
    print(f'orbital energies: {problem.orbital_energies}')

    print(f'{molecule}: {problem.num_spin_orbitals} spin-orbitals, {problem.num_alpha, problem.num_beta} electrons')

    num_orbitals = problem.num_spin_orbitals
    num_electrons = problem.num_alpha + problem.num_beta

    jw_mapper = JordanWignerMapper()
    bk_mapper = BravyiKitaevMapper()

    jw_hamiltonian, bk_hamiltonian = jw_mapper.map(fermionic_op), bk_mapper.map(fermionic_op)

    jw_list_op, bk_list_op = jw_hamiltonian.to_list(), bk_hamiltonian.to_list()

    print(f'num pauli terms: {len(jw_list_op), len(bk_list_op)}')

    with open(f'./{molecule}/JW_{molecule}_{basis}_{num_electrons}_electrons_{num_orbitals}_spin_orbitals_Hamiltonian_{len(jw_list_op)}_paulis.txt', 'w') as jwfile:

        print(f'Jordan-Wigner mapping of {molecule} Hamiltonian in {basis} basis.', file=jwfile)
        
        for op in jw_list_op:

            print(op[0], op[1], file=jwfile)

    with open(f'./{molecule}/BK_{molecule}_{basis}_{num_electrons}_electrons_{num_orbitals}_spin_orbitals_Hamiltonian_{len(bk_list_op)}_paulis.txt', 'w') as bkfile:

        print(f'Bravyi-Kitaev mapping of {molecule} Hamiltonian in {basis} basis.', file=bkfile)
        
        for op in bk_list_op:

            print(op[0], op[1], file=bkfile)

    print(bk_hamiltonian.equiv(jw_hamiltonian))

    return jw_hamiltonian, bk_hamiltonian

for active_space_transformer in [None]:

    get_hamiltonians(molecule="LiH", basis='sto3g', active_space_transformer=active_space_transformer)


for active_space_transformer in [None]:

    get_hamiltonians(molecule="BeH2", basis='sto3g', active_space_transformer=active_space_transformer)


for active_space_transformer in [None, ActiveSpaceTransformer(num_electrons=8, num_spatial_orbitals=6)]:

    get_hamiltonians(molecule="H2O", basis='sto3g', active_space_transformer=active_space_transformer)


for active_space_transformer in [None, ActiveSpaceTransformer(num_electrons=8, num_spatial_orbitals=7),
                                       ActiveSpaceTransformer(num_electrons=8, num_spatial_orbitals=6)]:
   
    get_hamiltonians(molecule="NH3", basis='sto3g', active_space_transformer=active_space_transformer)


for active_space_transformer in [None, ActiveSpaceTransformer(num_electrons=12, num_spatial_orbitals=9),
                                       ActiveSpaceTransformer(num_electrons=10, num_spatial_orbitals=8)]:

    get_hamiltonians(molecule="N2", basis='sto3g', active_space_transformer=active_space_transformer)


for active_space_transformer in [None,
                                 ActiveSpaceTransformer(num_electrons=12, num_spatial_orbitals=9),
                                 ActiveSpaceTransformer(num_electrons=10, num_spatial_orbitals=8)]:

    get_hamiltonians(molecule="C2", basis='sto3g', active_space_transformer=active_space_transformer)


for active_space_transformer in [None,
                                       ActiveSpaceTransformer(num_electrons=14, num_spatial_orbitals=12),
                                       ActiveSpaceTransformer(num_electrons=12, num_spatial_orbitals=10),
                                       ActiveSpaceTransformer(num_electrons=10, num_spatial_orbitals=8)]:

    get_hamiltonians(molecule="ethylene", basis='sto3g', active_space_transformer=active_space_transformer)


for active_space_transformer in [None,
                                 ActiveSpaceTransformer(num_electrons=42, num_spatial_orbitals=36),
                                 ActiveSpaceTransformer(num_electrons=40, num_spatial_orbitals=35),
                                 ActiveSpaceTransformer(num_electrons=38, num_spatial_orbitals=34),
                                 ActiveSpaceTransformer(num_electrons=36, num_spatial_orbitals=33),
                                 ActiveSpaceTransformer(num_electrons=34, num_spatial_orbitals=32),
                                 ActiveSpaceTransformer(num_electrons=32, num_spatial_orbitals=31),
                                 ActiveSpaceTransformer(num_electrons=30, num_spatial_orbitals=30),
                                 ActiveSpaceTransformer(num_electrons=28, num_spatial_orbitals=29),
                                 ActiveSpaceTransformer(num_electrons=26, num_spatial_orbitals=28),
                                 ActiveSpaceTransformer(num_electrons=24, num_spatial_orbitals=27),
                                 ActiveSpaceTransformer(num_electrons=22, num_spatial_orbitals=26),
                                 ActiveSpaceTransformer(num_electrons=20, num_spatial_orbitals=25),
                                 ActiveSpaceTransformer(num_electrons=18, num_spatial_orbitals=24),
                                 ActiveSpaceTransformer(num_electrons=16, num_spatial_orbitals=23),
                                 ActiveSpaceTransformer(num_electrons=14, num_spatial_orbitals=22),
                                 ActiveSpaceTransformer(num_electrons=10, num_spatial_orbitals=20)]:

    get_hamiltonians(molecule="benzene", basis='sto3g', active_space_transformer=active_space_transformer)










