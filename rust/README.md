# Bevy/Rust-Port des Hybrid-Meshers

Portierung des Hybrid-Meshers aus Blocky- und SDF-Anteil von C++/Godot nach
Rust/Bevy. Plan in `../doc/source/bevy_port_plan.md`, Fortschritt in
`../doc/source/bevy_port_todo.md`.

## Crates

| Crate | Abhaengigkeiten | Inhalt |
|---|---|---|
| `voxel-core` | `glam` | `VoxelBuffer`, Kanaele, Tiefen, SDF-Quantisierung, Mixel4, cube_tables, `Box3i`, SDF-Edits, Raycast, Generator-Trait |
| `voxel-mesher` | `glam`, `voxel-core` | Transvoxel, Transvoxel-Materialien, Blocky-Backdaten, Fluid-Modelle, Bibliotheks-Loader, Hybrid-Logik |
| `bevy-voxel` | `bevy`, beide obigen | Mesh-Erzeugung, WGSL-Material, Chunkverwaltung |

`voxel-core` und `voxel-mesher` haengen weder von Bevy noch von Godot ab.

## Verifizierte Versionen

- `cargo 1.99.1`, `rustc 1.99.1` (`cargo --version`, `rustc --version`)
- `bevy 0.20.1`, MIT OR Apache-2.0, rust-version 1.97.1 (`cargo info bevy`)
- `glam 0.33`, angeglichen an `bevy_math 0.20.1` (`cargo tree -i glam@0.33.12`)

## Bauen und testen

```sh
cargo test --workspace
cargo clippy --workspace --all-targets
cargo run -p bevy-voxel --example terrain
```

## Erzeugte Dateien

`voxel-mesher/src/transvoxel/tables.rs` wird maschinell aus
`../meshers/transvoxel/transvoxel_tables.cpp` erzeugt:

```sh
python3 -I tools/convert_transvoxel_tables.py \
    ../meshers/transvoxel/transvoxel_tables.cpp \
    voxel-mesher/src/transvoxel/tables.rs
```

## Lizenz

MIT, abgeleitet aus `../LICENSE.md`, Copyright 2016-2024 Marc Gilleron.
Die Transvoxel-Nachschlagetabellen tragen zusaetzlich den Hinweis von Eric
Lengyel, siehe Kopf von `voxel-mesher/src/transvoxel/tables.rs`.
