# Correctness

## Compile/Eager Fullgraph Status

| signature | operation | status | mode | compile | fullgraph | finite | compile abs | compile rel | gate | skip | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cl(2,0,0) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | vector_gp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | bivector_vector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | bivector_vector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | bivector_bivector_commutator | skipped | not_run | True | True | True |  |  | True | structural_empty_output:bivector_bivector_commutator_requires_n>=3 |  |
| Cl(2,0,0) | bivector_exp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | bivector_exp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | norm_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | norm_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | reverse_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | reverse_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | grade_involution_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | grade_involution_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | dual_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | dual_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | versor_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,0) | versor_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | vector_gp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | bivector_vector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | bivector_vector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | bivector_bivector_commutator | skipped | not_run | True | True | True |  |  | True | structural_empty_output:bivector_bivector_commutator_requires_n>=3 |  |
| Cl(1,1,0) | bivector_exp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | bivector_exp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | norm_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | norm_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | reverse_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | reverse_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | grade_involution_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | grade_involution_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | dual_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | dual_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | versor_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,1,0) | versor_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | vector_gp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | bivector_vector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | bivector_vector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | bivector_bivector_commutator | skipped | not_run | True | True | True |  |  | True | structural_empty_output:bivector_bivector_commutator_requires_n>=3 |  |
| Cl(1,0,1) | bivector_exp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | bivector_exp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | norm_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | norm_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | reverse_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | reverse_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | grade_involution_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | grade_involution_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | dual_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | dual_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | versor_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(1,0,1) | versor_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | vector_gp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | bivector_vector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | bivector_vector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | bivector_bivector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | bivector_bivector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | bivector_exp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | bivector_exp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | norm_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | norm_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | reverse_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | reverse_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | grade_involution_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | grade_involution_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | dual_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | dual_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | versor_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,0) | versor_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | vector_gp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | bivector_vector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | bivector_vector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | bivector_bivector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | bivector_bivector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | bivector_exp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | bivector_exp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | norm_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | norm_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | reverse_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | reverse_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | grade_involution_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | grade_involution_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | dual_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | dual_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | versor_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,1,0) | versor_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | vector_gp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | bivector_vector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | bivector_vector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | bivector_bivector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | bivector_bivector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | bivector_exp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | bivector_exp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | norm_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | norm_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | reverse_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | reverse_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | grade_involution_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | grade_involution_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | dual_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | dual_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | versor_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(2,0,1) | versor_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | vector_gp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | bivector_vector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | bivector_vector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | bivector_bivector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | bivector_bivector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | bivector_exp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | bivector_exp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | norm_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | norm_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | reverse_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | reverse_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | grade_involution_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | grade_involution_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | dual_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | dual_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | versor_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(4,0,0) | versor_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | vector_gp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | bivector_vector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | bivector_vector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | bivector_bivector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | bivector_bivector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | bivector_exp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | bivector_exp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | norm_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | norm_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | reverse_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | reverse_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | grade_involution_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | grade_involution_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | dual_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | dual_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | versor_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,1,0) | versor_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | vector_gp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | bivector_vector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | bivector_vector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | bivector_bivector_commutator | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | bivector_bivector_commutator | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | bivector_exp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | bivector_exp | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | norm_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | norm_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | reverse_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | reverse_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | grade_involution_default | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | grade_involution_default | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | dual_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | dual_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | versor_vector | ok | eager | True | True | True | 0 | 0 | True |  |  |
| Cl(3,0,1) | versor_vector | ok | aot_eager | True | True | True | 0 | 0 | True |  |  |
| Cl(5,0,0) | vector_gp | ok | eager | True | True | True | 0 | 0 | True |  |  |
| 3130 more rows in artifacts |  |  |  |  |  |  |  |  |  |  |  |

## Accumulated Rotor-Chain Drift

| signature | dtype | step | steps | angle | elapsed ms | drift abs | drift rel | norm drift abs | finite | gate | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cl(2,0,0) | float32 | 1 | 512 | 0.015 | 0.155333 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(2,0,0) | float32 | 2 | 512 | 0.015 | 0.241208 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,0) | float32 | 3 | 512 | 0.015 | 0.3165 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,0) | float32 | 5 | 512 | 0.015 | 0.404291 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,0) | float32 | 8 | 512 | 0.015 | 0.507416 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(2,0,0) | float32 | 12 | 512 | 0.015 | 0.623708 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(2,0,0) | float32 | 18 | 512 | 0.015 | 0.789041 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(2,0,0) | float32 | 28 | 512 | 0.015 | 0.989625 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(2,0,0) | float32 | 42 | 512 | 0.015 | 1.263 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(2,0,0) | float32 | 64 | 512 | 0.015 | 1.66204 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(2,0,0) | float32 | 97 | 512 | 0.015 | 2.13817 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(2,0,0) | float32 | 147 | 512 | 0.015 | 2.81567 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(2,0,0) | float32 | 223 | 512 | 0.015 | 3.87429 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(2,0,0) | float32 | 338 | 512 | 0.015 | 5.38417 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(2,0,0) | float32 | 512 | 512 | 0.015 | 7.59579 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(1,1,0) | float32 |  | 512 | 0.015 | 0 | 0 | 0 | 0 | True | True | skipped:requires_two_positive_axes_for_euclidean_plane_rotor |
| Cl(1,0,1) | float32 |  | 512 | 0.015 | 0 | 0 | 0 | 0 | True | True | skipped:requires_two_positive_axes_for_euclidean_plane_rotor |
| Cl(3,0,0) | float32 | 1 | 512 | 0.015 | 0.13025 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(3,0,0) | float32 | 2 | 512 | 0.015 | 0.220083 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,0) | float32 | 3 | 512 | 0.015 | 0.301708 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,0) | float32 | 5 | 512 | 0.015 | 0.393708 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,0) | float32 | 8 | 512 | 0.015 | 0.496333 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(3,0,0) | float32 | 12 | 512 | 0.015 | 0.610708 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(3,0,0) | float32 | 18 | 512 | 0.015 | 0.752375 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(3,0,0) | float32 | 28 | 512 | 0.015 | 0.946 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(3,0,0) | float32 | 42 | 512 | 0.015 | 1.19129 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(3,0,0) | float32 | 64 | 512 | 0.015 | 1.55046 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(3,0,0) | float32 | 97 | 512 | 0.015 | 2.11208 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(3,0,0) | float32 | 147 | 512 | 0.015 | 2.85492 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(3,0,0) | float32 | 223 | 512 | 0.015 | 3.93942 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(3,0,0) | float32 | 338 | 512 | 0.015 | 5.50788 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(3,0,0) | float32 | 512 | 512 | 0.015 | 7.81583 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(2,1,0) | float32 | 1 | 512 | 0.015 | 0.110917 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(2,1,0) | float32 | 2 | 512 | 0.015 | 0.19125 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(2,1,0) | float32 | 3 | 512 | 0.015 | 0.268042 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(2,1,0) | float32 | 5 | 512 | 0.015 | 0.35675 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(2,1,0) | float32 | 8 | 512 | 0.015 | 0.458958 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(2,1,0) | float32 | 12 | 512 | 0.015 | 0.574167 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(2,1,0) | float32 | 18 | 512 | 0.015 | 0.714958 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(2,1,0) | float32 | 28 | 512 | 0.015 | 0.907208 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(2,1,0) | float32 | 42 | 512 | 0.015 | 1.15192 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(2,1,0) | float32 | 64 | 512 | 0.015 | 1.50129 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(2,1,0) | float32 | 97 | 512 | 0.015 | 1.98979 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(2,1,0) | float32 | 147 | 512 | 0.015 | 2.71433 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(2,1,0) | float32 | 223 | 512 | 0.015 | 3.74833 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(2,1,0) | float32 | 338 | 512 | 0.015 | 5.36717 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(2,1,0) | float32 | 512 | 512 | 0.015 | 7.65304 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(2,0,1) | float32 | 1 | 512 | 0.015 | 0.107916 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(2,0,1) | float32 | 2 | 512 | 0.015 | 0.189583 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,1) | float32 | 3 | 512 | 0.015 | 0.265375 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,1) | float32 | 5 | 512 | 0.015 | 0.352458 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,1) | float32 | 8 | 512 | 0.015 | 0.451541 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(2,0,1) | float32 | 12 | 512 | 0.015 | 0.562583 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(2,0,1) | float32 | 18 | 512 | 0.015 | 0.699333 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(2,0,1) | float32 | 28 | 512 | 0.015 | 0.888041 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(2,0,1) | float32 | 42 | 512 | 0.015 | 1.12458 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(2,0,1) | float32 | 64 | 512 | 0.015 | 1.48367 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(2,0,1) | float32 | 97 | 512 | 0.015 | 1.95783 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(2,0,1) | float32 | 147 | 512 | 0.015 | 2.64992 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(2,0,1) | float32 | 223 | 512 | 0.015 | 3.66638 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(2,0,1) | float32 | 338 | 512 | 0.015 | 5.20129 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(2,0,1) | float32 | 512 | 512 | 0.015 | 7.47142 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(4,0,0) | float32 | 1 | 512 | 0.015 | 0.139417 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(4,0,0) | float32 | 2 | 512 | 0.015 | 0.226917 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,0) | float32 | 3 | 512 | 0.015 | 0.307834 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,0) | float32 | 5 | 512 | 0.015 | 0.40325 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,0) | float32 | 8 | 512 | 0.015 | 0.519542 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(4,0,0) | float32 | 12 | 512 | 0.015 | 0.654209 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(4,0,0) | float32 | 18 | 512 | 0.015 | 0.84625 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(4,0,0) | float32 | 28 | 512 | 0.015 | 1.05617 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(4,0,0) | float32 | 42 | 512 | 0.015 | 1.32133 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(4,0,0) | float32 | 64 | 512 | 0.015 | 1.697 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(4,0,0) | float32 | 97 | 512 | 0.015 | 2.22287 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(4,0,0) | float32 | 147 | 512 | 0.015 | 2.97837 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(4,0,0) | float32 | 223 | 512 | 0.015 | 4.12104 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(4,0,0) | float32 | 338 | 512 | 0.015 | 5.80517 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(4,0,0) | float32 | 512 | 512 | 0.015 | 8.28133 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(3,1,0) | float32 | 1 | 512 | 0.015 | 0.111542 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(3,1,0) | float32 | 2 | 512 | 0.015 | 0.196959 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(3,1,0) | float32 | 3 | 512 | 0.015 | 0.286875 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(3,1,0) | float32 | 5 | 512 | 0.015 | 0.381584 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(3,1,0) | float32 | 8 | 512 | 0.015 | 0.488917 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(3,1,0) | float32 | 12 | 512 | 0.015 | 0.612125 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(3,1,0) | float32 | 18 | 512 | 0.015 | 0.761667 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(3,1,0) | float32 | 28 | 512 | 0.015 | 0.966042 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(3,1,0) | float32 | 42 | 512 | 0.015 | 1.24017 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(3,1,0) | float32 | 64 | 512 | 0.015 | 1.61846 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(3,1,0) | float32 | 97 | 512 | 0.015 | 2.14896 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(3,1,0) | float32 | 147 | 512 | 0.015 | 2.90996 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(3,1,0) | float32 | 223 | 512 | 0.015 | 4.04817 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(3,1,0) | float32 | 338 | 512 | 0.015 | 5.71746 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(3,1,0) | float32 | 512 | 512 | 0.015 | 8.20742 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(3,0,1) | float32 | 1 | 512 | 0.015 | 0.109583 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(3,0,1) | float32 | 2 | 512 | 0.015 | 0.1935 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,1) | float32 | 3 | 512 | 0.015 | 0.272875 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,1) | float32 | 5 | 512 | 0.015 | 0.365416 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,1) | float32 | 8 | 512 | 0.015 | 0.471833 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(3,0,1) | float32 | 12 | 512 | 0.015 | 0.622708 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(3,0,1) | float32 | 18 | 512 | 0.015 | 0.780208 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(3,0,1) | float32 | 28 | 512 | 0.015 | 0.983041 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(3,0,1) | float32 | 42 | 512 | 0.015 | 1.23646 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(3,0,1) | float32 | 64 | 512 | 0.015 | 1.59787 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(3,0,1) | float32 | 97 | 512 | 0.015 | 2.1085 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(3,0,1) | float32 | 147 | 512 | 0.015 | 2.8435 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(3,0,1) | float32 | 223 | 512 | 0.015 | 3.93058 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(3,0,1) | float32 | 338 | 512 | 0.015 | 5.54579 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(3,0,1) | float32 | 512 | 512 | 0.015 | 7.94054 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(5,0,0) | float32 | 1 | 512 | 0.015 | 0.126792 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(5,0,0) | float32 | 2 | 512 | 0.015 | 0.221917 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(5,0,0) | float32 | 3 | 512 | 0.015 | 0.312334 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(5,0,0) | float32 | 5 | 512 | 0.015 | 0.418209 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(5,0,0) | float32 | 8 | 512 | 0.015 | 0.539959 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(5,0,0) | float32 | 12 | 512 | 0.015 | 0.678167 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(5,0,0) | float32 | 18 | 512 | 0.015 | 0.913417 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(5,0,0) | float32 | 28 | 512 | 0.015 | 1.1575 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(5,0,0) | float32 | 42 | 512 | 0.015 | 1.45713 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(5,0,0) | float32 | 64 | 512 | 0.015 | 1.88254 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(5,0,0) | float32 | 97 | 512 | 0.015 | 2.48225 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(5,0,0) | float32 | 147 | 512 | 0.015 | 3.37154 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(5,0,0) | float32 | 223 | 512 | 0.015 | 4.66633 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(5,0,0) | float32 | 338 | 512 | 0.015 | 6.58692 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(5,0,0) | float32 | 512 | 512 | 0.015 | 9.43158 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(4,1,0) | float32 | 1 | 512 | 0.015 | 0.125583 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(4,1,0) | float32 | 2 | 512 | 0.015 | 0.221 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(4,1,0) | float32 | 3 | 512 | 0.015 | 0.311625 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(4,1,0) | float32 | 5 | 512 | 0.015 | 0.418125 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(4,1,0) | float32 | 8 | 512 | 0.015 | 0.545041 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(4,1,0) | float32 | 12 | 512 | 0.015 | 0.683541 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(4,1,0) | float32 | 18 | 512 | 0.015 | 0.85725 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(4,1,0) | float32 | 28 | 512 | 0.015 | 1.0925 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(4,1,0) | float32 | 42 | 512 | 0.015 | 1.39087 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(4,1,0) | float32 | 64 | 512 | 0.015 | 1.83017 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(4,1,0) | float32 | 97 | 512 | 0.015 | 2.43175 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(4,1,0) | float32 | 147 | 512 | 0.015 | 3.311 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(4,1,0) | float32 | 223 | 512 | 0.015 | 4.60017 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(4,1,0) | float32 | 338 | 512 | 0.015 | 6.49946 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(4,1,0) | float32 | 512 | 512 | 0.015 | 9.37546 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(4,0,1) | float32 | 1 | 512 | 0.015 | 0.14325 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(4,0,1) | float32 | 2 | 512 | 0.015 | 0.234584 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,1) | float32 | 3 | 512 | 0.015 | 0.322167 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,1) | float32 | 5 | 512 | 0.015 | 0.42425 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,1) | float32 | 8 | 512 | 0.015 | 0.543 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(4,0,1) | float32 | 12 | 512 | 0.015 | 0.677084 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(4,0,1) | float32 | 18 | 512 | 0.015 | 0.840417 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(4,0,1) | float32 | 28 | 512 | 0.015 | 1.06712 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(4,0,1) | float32 | 42 | 512 | 0.015 | 1.35492 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(4,0,1) | float32 | 64 | 512 | 0.015 | 1.76883 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(4,0,1) | float32 | 97 | 512 | 0.015 | 2.34804 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(4,0,1) | float32 | 147 | 512 | 0.015 | 3.18788 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(4,0,1) | float32 | 223 | 512 | 0.015 | 4.43042 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(4,0,1) | float32 | 338 | 512 | 0.015 | 6.26292 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(4,0,1) | float32 | 512 | 512 | 0.015 | 9.01721 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(6,0,0) | float32 | 1 | 512 | 0.015 | 0.136625 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(6,0,0) | float32 | 2 | 512 | 0.015 | 0.243625 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(6,0,0) | float32 | 3 | 512 | 0.015 | 0.345042 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(6,0,0) | float32 | 5 | 512 | 0.015 | 0.4655 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(6,0,0) | float32 | 8 | 512 | 0.015 | 0.604042 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(6,0,0) | float32 | 12 | 512 | 0.015 | 0.762584 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(6,0,0) | float32 | 18 | 512 | 0.015 | 0.958709 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(6,0,0) | float32 | 28 | 512 | 0.015 | 1.23221 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| 2602 more rows in artifacts |  |  |  |  |  |  |  |  |  |  |  |
