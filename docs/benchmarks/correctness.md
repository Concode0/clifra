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
| Cl(2,0,0) | float32 | 1 | 512 | 0.015 | 0.148416 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(2,0,0) | float32 | 2 | 512 | 0.015 | 0.236375 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,0) | float32 | 3 | 512 | 0.015 | 0.313833 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,0) | float32 | 5 | 512 | 0.015 | 0.404541 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,0) | float32 | 8 | 512 | 0.015 | 0.515916 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(2,0,0) | float32 | 12 | 512 | 0.015 | 0.634708 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(2,0,0) | float32 | 18 | 512 | 0.015 | 0.77425 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(2,0,0) | float32 | 28 | 512 | 0.015 | 0.962625 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(2,0,0) | float32 | 42 | 512 | 0.015 | 1.20254 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(2,0,0) | float32 | 64 | 512 | 0.015 | 1.54113 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(2,0,0) | float32 | 97 | 512 | 0.015 | 2.01971 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(2,0,0) | float32 | 147 | 512 | 0.015 | 2.70921 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(2,0,0) | float32 | 223 | 512 | 0.015 | 3.70883 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(2,0,0) | float32 | 338 | 512 | 0.015 | 5.20038 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(2,0,0) | float32 | 512 | 512 | 0.015 | 7.65304 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(1,1,0) | float32 |  | 512 | 0.015 | 0 | 0 | 0 | 0 | True | True | skipped:requires_two_positive_axes_for_euclidean_plane_rotor |
| Cl(1,0,1) | float32 |  | 512 | 0.015 | 0 | 0 | 0 | 0 | True | True | skipped:requires_two_positive_axes_for_euclidean_plane_rotor |
| Cl(3,0,0) | float32 | 1 | 512 | 0.015 | 0.118542 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(3,0,0) | float32 | 2 | 512 | 0.015 | 0.205292 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,0) | float32 | 3 | 512 | 0.015 | 0.291917 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,0) | float32 | 5 | 512 | 0.015 | 0.383375 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,0) | float32 | 8 | 512 | 0.015 | 0.49675 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(3,0,0) | float32 | 12 | 512 | 0.015 | 0.61525 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(3,0,0) | float32 | 18 | 512 | 0.015 | 0.758583 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(3,0,0) | float32 | 28 | 512 | 0.015 | 0.956875 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(3,0,0) | float32 | 42 | 512 | 0.015 | 1.20692 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(3,0,0) | float32 | 64 | 512 | 0.015 | 1.56038 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(3,0,0) | float32 | 97 | 512 | 0.015 | 2.05179 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(3,0,0) | float32 | 147 | 512 | 0.015 | 2.76375 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(3,0,0) | float32 | 223 | 512 | 0.015 | 3.80542 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(3,0,0) | float32 | 338 | 512 | 0.015 | 5.40258 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(3,0,0) | float32 | 512 | 512 | 0.015 | 7.75396 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(2,1,0) | float32 | 1 | 512 | 0.015 | 0.137125 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(2,1,0) | float32 | 2 | 512 | 0.015 | 0.227583 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(2,1,0) | float32 | 3 | 512 | 0.015 | 0.308667 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(2,1,0) | float32 | 5 | 512 | 0.015 | 0.400417 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(2,1,0) | float32 | 8 | 512 | 0.015 | 0.509875 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(2,1,0) | float32 | 12 | 512 | 0.015 | 0.62625 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(2,1,0) | float32 | 18 | 512 | 0.015 | 0.769083 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(2,1,0) | float32 | 28 | 512 | 0.015 | 0.963292 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(2,1,0) | float32 | 42 | 512 | 0.015 | 1.20962 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(2,1,0) | float32 | 64 | 512 | 0.015 | 1.56279 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(2,1,0) | float32 | 97 | 512 | 0.015 | 2.05633 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(2,1,0) | float32 | 147 | 512 | 0.015 | 2.77929 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(2,1,0) | float32 | 223 | 512 | 0.015 | 4.11333 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(2,1,0) | float32 | 338 | 512 | 0.015 | 5.75533 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(2,1,0) | float32 | 512 | 512 | 0.015 | 8.14433 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(2,0,1) | float32 | 1 | 512 | 0.015 | 0.112459 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(2,0,1) | float32 | 2 | 512 | 0.015 | 0.195417 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,1) | float32 | 3 | 512 | 0.015 | 0.272834 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,1) | float32 | 5 | 512 | 0.015 | 0.361209 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(2,0,1) | float32 | 8 | 512 | 0.015 | 0.460875 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(2,0,1) | float32 | 12 | 512 | 0.015 | 0.577542 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(2,0,1) | float32 | 18 | 512 | 0.015 | 0.716584 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(2,0,1) | float32 | 28 | 512 | 0.015 | 0.905542 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(2,0,1) | float32 | 42 | 512 | 0.015 | 1.14833 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(2,0,1) | float32 | 64 | 512 | 0.015 | 1.50146 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(2,0,1) | float32 | 97 | 512 | 0.015 | 1.98829 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(2,0,1) | float32 | 147 | 512 | 0.015 | 2.68471 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(2,0,1) | float32 | 223 | 512 | 0.015 | 3.71258 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(2,0,1) | float32 | 338 | 512 | 0.015 | 5.3645 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(2,0,1) | float32 | 512 | 512 | 0.015 | 7.70454 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(4,0,0) | float32 | 1 | 512 | 0.015 | 0.1565 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(4,0,0) | float32 | 2 | 512 | 0.015 | 0.243959 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,0) | float32 | 3 | 512 | 0.015 | 0.3265 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,0) | float32 | 5 | 512 | 0.015 | 0.430417 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,0) | float32 | 8 | 512 | 0.015 | 0.542084 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(4,0,0) | float32 | 12 | 512 | 0.015 | 0.666042 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(4,0,0) | float32 | 18 | 512 | 0.015 | 0.820292 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(4,0,0) | float32 | 28 | 512 | 0.015 | 1.03104 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(4,0,0) | float32 | 42 | 512 | 0.015 | 1.29996 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(4,0,0) | float32 | 64 | 512 | 0.015 | 1.68429 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(4,0,0) | float32 | 97 | 512 | 0.015 | 2.23137 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(4,0,0) | float32 | 147 | 512 | 0.015 | 3.02233 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(4,0,0) | float32 | 223 | 512 | 0.015 | 4.32688 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(4,0,0) | float32 | 338 | 512 | 0.015 | 6.10529 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(4,0,0) | float32 | 512 | 512 | 0.015 | 8.70325 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(3,1,0) | float32 | 1 | 512 | 0.015 | 0.116167 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(3,1,0) | float32 | 2 | 512 | 0.015 | 0.203458 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(3,1,0) | float32 | 3 | 512 | 0.015 | 0.285875 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(3,1,0) | float32 | 5 | 512 | 0.015 | 0.381667 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(3,1,0) | float32 | 8 | 512 | 0.015 | 0.491042 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(3,1,0) | float32 | 12 | 512 | 0.015 | 0.616458 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(3,1,0) | float32 | 18 | 512 | 0.015 | 0.768708 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(3,1,0) | float32 | 28 | 512 | 0.015 | 0.983083 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(3,1,0) | float32 | 42 | 512 | 0.015 | 1.25208 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(3,1,0) | float32 | 64 | 512 | 0.015 | 1.63592 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(3,1,0) | float32 | 97 | 512 | 0.015 | 2.17729 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(3,1,0) | float32 | 147 | 512 | 0.015 | 3.10425 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(3,1,0) | float32 | 223 | 512 | 0.015 | 4.30017 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(3,1,0) | float32 | 338 | 512 | 0.015 | 6.18804 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(3,1,0) | float32 | 512 | 512 | 0.015 | 8.77854 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(3,0,1) | float32 | 1 | 512 | 0.015 | 0.119125 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(3,0,1) | float32 | 2 | 512 | 0.015 | 0.27925 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,1) | float32 | 3 | 512 | 0.015 | 0.365208 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,1) | float32 | 5 | 512 | 0.015 | 0.462583 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(3,0,1) | float32 | 8 | 512 | 0.015 | 0.573125 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(3,0,1) | float32 | 12 | 512 | 0.015 | 0.695625 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(3,0,1) | float32 | 18 | 512 | 0.015 | 0.845667 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(3,0,1) | float32 | 28 | 512 | 0.015 | 1.05329 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(3,0,1) | float32 | 42 | 512 | 0.015 | 1.31612 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(3,0,1) | float32 | 64 | 512 | 0.015 | 1.69262 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(3,0,1) | float32 | 97 | 512 | 0.015 | 2.23075 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(3,0,1) | float32 | 147 | 512 | 0.015 | 3.03437 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(3,0,1) | float32 | 223 | 512 | 0.015 | 4.24604 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(3,0,1) | float32 | 338 | 512 | 0.015 | 5.94846 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(3,0,1) | float32 | 512 | 512 | 0.015 | 8.40904 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(5,0,0) | float32 | 1 | 512 | 0.015 | 0.16975 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(5,0,0) | float32 | 2 | 512 | 0.015 | 0.277125 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(5,0,0) | float32 | 3 | 512 | 0.015 | 0.373333 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(5,0,0) | float32 | 5 | 512 | 0.015 | 0.493125 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(5,0,0) | float32 | 8 | 512 | 0.015 | 0.620208 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(5,0,0) | float32 | 12 | 512 | 0.015 | 0.771167 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(5,0,0) | float32 | 18 | 512 | 0.015 | 0.953458 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(5,0,0) | float32 | 28 | 512 | 0.015 | 1.20492 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(5,0,0) | float32 | 42 | 512 | 0.015 | 1.51042 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(5,0,0) | float32 | 64 | 512 | 0.015 | 1.95896 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(5,0,0) | float32 | 97 | 512 | 0.015 | 2.59987 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(5,0,0) | float32 | 147 | 512 | 0.015 | 3.49967 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(5,0,0) | float32 | 223 | 512 | 0.015 | 4.81554 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(5,0,0) | float32 | 338 | 512 | 0.015 | 6.89658 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(5,0,0) | float32 | 512 | 512 | 0.015 | 9.94896 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(4,1,0) | float32 | 1 | 512 | 0.015 | 0.133791 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(4,1,0) | float32 | 2 | 512 | 0.015 | 0.234416 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(4,1,0) | float32 | 3 | 512 | 0.015 | 0.328458 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(4,1,0) | float32 | 5 | 512 | 0.015 | 0.43775 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(4,1,0) | float32 | 8 | 512 | 0.015 | 0.563625 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(4,1,0) | float32 | 12 | 512 | 0.015 | 0.705333 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(4,1,0) | float32 | 18 | 512 | 0.015 | 0.87875 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(4,1,0) | float32 | 28 | 512 | 0.015 | 1.11629 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(4,1,0) | float32 | 42 | 512 | 0.015 | 1.41787 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(4,1,0) | float32 | 64 | 512 | 0.015 | 1.85008 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(4,1,0) | float32 | 97 | 512 | 0.015 | 2.58521 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(4,1,0) | float32 | 147 | 512 | 0.015 | 3.49217 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(4,1,0) | float32 | 223 | 512 | 0.015 | 4.80163 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(4,1,0) | float32 | 338 | 512 | 0.015 | 6.75213 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(4,1,0) | float32 | 512 | 512 | 0.015 | 9.59717 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(4,0,1) | float32 | 1 | 512 | 0.015 | 0.197333 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(4,0,1) | float32 | 2 | 512 | 0.015 | 0.299917 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,1) | float32 | 3 | 512 | 0.015 | 0.389792 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,1) | float32 | 5 | 512 | 0.015 | 0.504583 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(4,0,1) | float32 | 8 | 512 | 0.015 | 0.624667 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(4,0,1) | float32 | 12 | 512 | 0.015 | 0.762208 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(4,0,1) | float32 | 18 | 512 | 0.015 | 0.959792 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(4,0,1) | float32 | 28 | 512 | 0.015 | 1.19096 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| Cl(4,0,1) | float32 | 42 | 512 | 0.015 | 1.4815 | 7.665e-07 | 9.486e-07 | 2.027e-06 | True | True |  |
| Cl(4,0,1) | float32 | 64 | 512 | 0.015 | 1.90592 | 1.007e-06 | 1.229e-06 | 2.086e-06 | True | True |  |
| Cl(4,0,1) | float32 | 97 | 512 | 0.015 | 2.50346 | 1.583e-06 | 1.593e-06 | 3.099e-06 | True | True |  |
| Cl(4,0,1) | float32 | 147 | 512 | 0.015 | 3.37796 | 1.574e-06 | 1.954e-06 | 4.053e-06 | True | True |  |
| Cl(4,0,1) | float32 | 223 | 512 | 0.015 | 4.62721 | 3.168e-06 | 3.235e-06 | 6.557e-06 | True | True |  |
| Cl(4,0,1) | float32 | 338 | 512 | 0.015 | 6.47529 | 4.807e-06 | 5.132e-06 | 1.043e-05 | True | True |  |
| Cl(4,0,1) | float32 | 512 | 512 | 0.015 | 9.48458 | 8.462e-06 | 8.592e-06 | 1.711e-05 | True | True |  |
| Cl(6,0,0) | float32 | 1 | 512 | 0.015 | 0.14875 | 3.568e-08 | 3.568e-08 | 5.960e-08 | True | True |  |
| Cl(6,0,0) | float32 | 2 | 512 | 0.015 | 0.253458 | 4.882e-08 | 4.884e-08 | 1.192e-07 | True | True |  |
| Cl(6,0,0) | float32 | 3 | 512 | 0.015 | 0.352333 | 5.574e-08 | 5.580e-08 | 1.192e-07 | True | True |  |
| Cl(6,0,0) | float32 | 5 | 512 | 0.015 | 0.469292 | 7.118e-08 | 7.138e-08 | 1.192e-07 | True | True |  |
| Cl(6,0,0) | float32 | 8 | 512 | 0.015 | 0.604958 | 1.151e-07 | 1.159e-07 | 2.384e-07 | True | True |  |
| Cl(6,0,0) | float32 | 12 | 512 | 0.015 | 0.76025 | 2.470e-07 | 2.511e-07 | 5.364e-07 | True | True |  |
| Cl(6,0,0) | float32 | 18 | 512 | 0.015 | 0.962458 | 3.280e-07 | 3.403e-07 | 7.153e-07 | True | True |  |
| Cl(6,0,0) | float32 | 28 | 512 | 0.015 | 1.244 | 5.590e-07 | 6.122e-07 | 1.192e-06 | True | True |  |
| 2602 more rows in artifacts |  |  |  |  |  |  |  |  |  |  |  |
