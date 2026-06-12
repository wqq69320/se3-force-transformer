# Scientific Demo Safety

The demo scripts produce qualitative scientific-AI diagnostics:

- force visualization
- short structure relaxation
- short molecular rollout
- energy-drift curve

They should be described as demonstrations of learned equivariant force-field behavior, not proof of physical correctness.

## Commands

```bash
python3 scripts/demo_force_field_visualization.py --config configs/molecular/scale_synthetic_n8.yaml --checkpoint outputs/scale_synthetic_n8/best.pt --output outputs/demos/force_viz
python3 scripts/demo_structure_relaxation.py --config configs/molecular/scale_synthetic_n8.yaml --checkpoint outputs/scale_synthetic_n8/best.pt --output outputs/demos/relaxation
python3 scripts/demo_md_rollout.py --config configs/molecular/scale_synthetic_n8.yaml --checkpoint outputs/scale_synthetic_n8/best.pt --steps 50 --dt_fs 0.25 --output outputs/demos/md_rollout
```

## What Can Be Claimed

It is acceptable to say a demo shows interpretable predicted forces or a short diagnostic rollout.

It is not acceptable to claim stable molecular dynamics, transferable chemistry, or production-quality force fields without larger real-data validation.
