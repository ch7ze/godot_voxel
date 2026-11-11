# Umbau-Fortschritt: Godot/C++ nach Bevy/Rust

Plan siehe `bevy_port_plan.md`. Stand wird bei jedem Paket aktualisiert.

Verifizierte Werkzeuglage (2026-11-11):

- `cargo 1.99.1`, `rustc 1.99.1` lokal vorhanden (`cargo --version`, `rustc --version`)
- `bevy` Registry-Version `0.20.1`, MIT OR Apache-2.0, rust-version 1.97.1 (`cargo info bevy`)
- `transvoxel` Registry-Version `2.0.0`, MIT OR Apache-2.0 (`cargo info transvoxel`)
- Abweichung zum Plan, dort wurde gegen Bevy 0.18 recherchiert

## Pakete

- [x] P0 Workspace `rust/` mit drei Crates, Manifeste, `.gitignore`
- [x] P1 voxel-core, Kanaele, Tiefen, SDF-Quantisierung, ZXY-Indizierung, Mixel4, cube_tables
      Module `constants`, `funcs`, `channel`, `mixel4`, `cube`, `math`, `buffer`. 51 Tests gruen.
- [x] P2 voxel-mesher, Transvoxel-Geometrie und Tabellen
      Tabellen maschinell erzeugt via `rust/tools/convert_transvoxel_tables.py`.
      Regulaerer Pass und Transition-Pass portiert. 42 Tests gruen.
- [x] P3 voxel-mesher, Transvoxel-Materialien Mixel4 und Null
      `single_s2` und `single_s4` zurueckgestellt, siehe Plan Abschnitt 4 P3.
- [x] P4 voxel-mesher, Blocky-Backdaten und Modellpfad
      Module `baked_library`, `fluids`, `library_desc` (Loader statt Godot-Ressourcen),
      Modellpfad in `blocky/mod.rs`. 102 Tests gruen im Paket.
- [x] P5 voxel-mesher, Hybrid-Logik
      Neutrale `MeshOutput`-Struktur, Build-Reihenfolge nach Plan 1.5,
      `build_flattened_sdf`, Fluid-Pass auf DATA5, Modell-Materialindex 2+N.
      Abweichung bewusst, Fluid-Pass schreibt in eigenen Output statt
      Zustandssicherung. 137 Tests gruen im Paket.
- [x] P6 bevy-voxel, Mesh-Erzeugung und WGSL-Material
      Eigene Attribut-IDs ab 8 statt Godot-ARRAY_CUSTOM-Slots, verifiziert in
      `bevy_mesh-0.20.1/src/mesh.rs:407`. WGSL-Port des Godot-Shaders.
      Bevy-Zielversion 0.20.1, glam im Workspace auf 0.33 angeglichen.
- [x] P7 bevy-voxel, Chunkverwaltung und Nebenlaeufigkeit
      Modul `chunk` mit Chunk-Entities, LOD-Auswahl, Transition-Masken,
      gepadderter Pufferbereitstellung und Meshing auf `AsyncComputeTaskPool`.
      Eigenentwicklung, Begruendung im Modulkopf.
- [x] P8 voxel-core, SDF-Edits und Raycast
      Modul `sdf_ops` mit Kugel, Box, Punkt, `sdf_blend` und DDA-Raycast.
      Zwei Voxel Rand bei Kugel und Box wie in `edition/funcs.h:316-323`.
- [x] P9 Generator-Trait und Referenzgenerator
      Modul `generator` mit Trait `VoxelGenerator`, `GenerateResult` und
      `FlatGenerator` als Referenz.

## Offene Punkte aus Plan 5.3, Stand nach dem Umbau

- Bevy-Zielversion festgelegt auf `0.20.1`, verifiziert per `cargo info bevy`
- Chunkverwaltung als Eigenentwicklung umgesetzt. `bevy_voxel_world 0.19.3`
  nimmt ueber `chunk_meshing_delegate` einen Fremdmesher auf
  (`bevy_voxel_world-0.19.3/src/configuration.rs:43-57`, `:244-248`), uebergibt
  jedoch `VoxelArray<I>` mit einem Material-Index je Voxel, waehrend der
  Hybrid-Mesher SDF, TYPE, INDICES, WEIGHTS und DATA5 gleichzeitig liest
- `single_s2` und `single_s4` aus P3 nicht portiert, der `rame`-Generator
  schreibt Mixel4-Materialien
- Rauschbibliothek fuer P9 weiterhin offen, der Referenzgenerator ist
  rauschfrei
- Verhaeltnis zu `rame` weiterhin offen, die Edit-API ist frei gestaltet und
  bildet die gdext-Signaturen nicht nach

## Pruefstand

- `cargo test --workspace` 271 Tests gruen
- Beispiel `cargo run -p bevy-voxel --example terrain`
