# Portierungsplan: Hybrid-Mesher (Blocky + SDF) von C++/Godot nach Rust/Bevy

Stand 2026-11-10. Branch `hybrid-mesher`.

Alle Dateiangaben beziehen sich auf dieses Repository, alle Zeilenzahlen auf
`wc -l` zum oben genannten Stand.

> **Umsetzungsstand 2026-11-11.** Der Plan ist umgesetzt, der Code liegt unter
> `rust/`. Der Fortschritt je Paket samt Abweichungen steht in
> `bevy_port_todo.md`. Abweichungen gegenueber diesem Plan, jeweils dort
> begruendet:
>
> - Bevy-Zielversion `0.20.1` statt der hier recherchierten 0.18
> - `glam` im Workspace auf `0.33`, angeglichen an `bevy_math 0.20.1`
> - P2 als Eigenportierung umgesetzt, der Crate `transvoxel` wird nicht genutzt
> - P7 als Eigenentwicklung, `bevy_voxel_world` nimmt ueber
>   `chunk_meshing_delegate` zwar einen Fremdmesher auf, uebergibt jedoch nur
>   einen Material-Index je Voxel
> - P8 liegt in `voxel-core` statt in `bevy-voxel`, die Operationen sind
>   engine-neutral

---

## 1. Zielbestimmung und Ist-Stand

### 1.1 Was portiert werden soll

Laut Auftrag nur der Hybrid-Mesher für Blocky- und SDF-Terrain, nicht das
gesamte Modul. Der Hybrid-Mesher liegt in `meshers/hybrid/voxel_mesher_hybrid.h`
(179 Zeilen) und `meshers/hybrid/voxel_mesher_hybrid.cpp` (1083 Zeilen).

Dessen Dokumentation im Header (`voxel_mesher_hybrid.h:14-23`) beschreibt den
Aufbau als eine Chunk-Mesh mit zwei Anteilen, einer glatten Iso-Fläche aus dem
SDF-Kanal über Transvoxel und scharfkantigen Modellen aus dem TYPE-Kanal
analog `VoxelMesherBlocky`. Der glatte Anteil wird an eine intern gehaltene
`VoxelMesherTransvoxel`-Instanz delegiert (`voxel_mesher_hybrid.h:22-23`,
Delegation sichtbar in `voxel_mesher_hybrid.cpp:550-555`).

Daraus folgt der tatsächliche Portierungsumfang. Der Hybrid-Mesher allein ist
nicht lauffähig, weil er Transvoxel, die Blocky-Modell-Backdaten und
`VoxelBuffer` als Datenquelle voraussetzt.

### 1.2 Verwendung unter `../rame`

Geprüft wurde `/home/ubuntu/git/rame`. Es handelt sich um ein getrenntes
Projekt mit Godot-Projekt, Rust-GDExtension und einem C++-Generator.

Struktur laut Verzeichnis-Listing:

- `rame/godot/` Godot-Projekt mit `project.godot`, `rame.gdextension`,
  fünf Szenen unter `rame/godot/scenes/`.
- `rame/rust/` Cargo-Workspace mit zwei Membern, `game` und `scene_nodes`
  (`rame/rust/Cargo.toml`), wobei nur `game` per `default-members` gebaut wird.
- `rame/godot-voxel-cpp/terrain_generator/` ein C++-Generator
  (`generator.h`, `generator.cpp`).
- `rame/tools/vscode-rame/` eine VS-Code-Erweiterung.

Der Rust-Anteil umfasst insgesamt 863 Zeilen über 27 Dateien
(`find rust -name '*.rs' | xargs wc -l`). Es wird ausschliesslich `godot`
0.5.5 mit Feature `api-custom` verwendet (`rame/rust/game/Cargo.toml`), also
gdext als GDExtension-Binding. Bevy wird dort nicht verwendet.

Konkret genutzte Voxel-API in `rame/rust/game/src/voxel_edits/voxel_edits.rs`:

- `godot::classes::VoxelLodTerrain`, `VoxelToolLodTerrain`, `VoxelRaycastResult`
- `godot::classes::voxel_buffer::ChannelId` mit `CHANNEL_SDF`
- `godot::classes::voxel_tool::Mode` mit `ADD` und `REMOVE`
- Methoden `get_voxel_tool`, `raycast`, `set_channel`, `set_mode`,
  `do_sphere`, `do_box`, `do_point`

Die Szene `rame/godot/scenes/world.tscn` instanziiert als Subresourcen
`TerrainGenerator`, `VoxelMesherHybrid` und `VoxelFormat` und setzt diese an
einem `VoxelLodTerrain` mit `streaming_system = 1` und Skalierung 0.5.

Der C++-Generator `rame/godot-voxel-cpp/terrain_generator/generator.h` erbt
von `zylann::voxel::VoxelGeneratorHeightmap`, überschreibt `generate_block`
und `get_used_channels_mask` und nutzt `ZN_FastNoiseLite` sowie optional
`FastNoise2` (`VOXEL_ENABLE_FAST_NOISE_2`). Er liefert Biome, Höhlen und
Mixel4-Oberflächenmaterialien.

Konsequenz für den Plan. Ein Bevy-Port kann die gdext-Schicht von `rame`
nicht weiterverwenden, da `godot::classes::*` an die Godot-Laufzeit gebunden
ist. Die oben aufgezählte API-Oberfläche definiert jedoch exakt, welche
Funktionalität das Bevy-Zielsystem mindestens anbieten muss, nämlich SDF-Edits
mit Kugel, Box und Einzelvoxel sowie ein Raycast gegen das Terrain.

### 1.3 Abhängigkeitsanalyse des Mesher-Kerns

Zeilenumfang der unmittelbar betroffenen Dateien:

| Bereich | Dateien | Zeilen |
|---|---|---|
| Hybrid | `meshers/hybrid/*` | 1262 |
| Transvoxel | `meshers/transvoxel/*` | 4270 |
| Blocky | `meshers/blocky/*` | 5257 |
| Mesher-Basis | `meshers/voxel_mesher.{h,cpp}` | 368 |
| Storage | `storage/voxel_buffer.{h,cpp}`, `mixel4.h`, `funcs.h` | 2785 |
| Konstanten | `constants/cube_tables.*` | 346 |
| Shader | `shaders/voxel_mesher_hybrid_shader.h` | 51 |

Innerhalb von Transvoxel entfallen 1081 Zeilen auf reine Nachschlagetabellen
(`meshers/transvoxel/transvoxel_tables.cpp`).

Die Godot-Kopplung ist unterschiedlich stark. `meshers/transvoxel/transvoxel.h`
enthält 8 Treffer auf Godot-Typmuster, `transvoxel.cpp` 16
(`grep -c "Ref<\|Variant\|Array\|GDCLASS\|Object"`), bei zusammen 1757 Zeilen.
Der Transvoxel-Kern ist damit weitgehend engine-neutral. Dagegen sind
`VoxelMesher::Output::Surface` mit `Array arrays` und
`Mesh::PrimitiveType primitive_type` (`meshers/voxel_mesher.h:51-57`) direkt
an Godot gebunden und müssen ersetzt werden.

Die Liste der von den Meshern eingebundenen `util`-Header umfasst 54 Einträge,
davon 24 unter `util/godot/`. Letztere entfallen im Bevy-Port vollständig,
Ersatz liefern `bevy_math`, `bevy_mesh` und `bevy_asset`.

### 1.4 Datenmodell

`storage/voxel_buffer.h:40-48` definiert acht Kanäle, `CHANNEL_TYPE`,
`CHANNEL_SDF`, `CHANNEL_COLOR`, `CHANNEL_INDICES`, `CHANNEL_WEIGHTS`,
`CHANNEL_DATA5`, `CHANNEL_DATA6`, `CHANNEL_DATA7`. Tiefen sind 8, 16, 32 und
64 Bit (`voxel_buffer.h:63-68`).

Für den Hybrid-Mesher relevante Vorgaben:

- `CHANNEL_TYPE` Standardtiefe 16 Bit (`voxel_buffer.h:162`)
- `CHANNEL_SDF` Standardtiefe 16 Bit (`voxel_buffer.h:163`)
- `CHANNEL_INDICES` Standardtiefe 32 Bit, laut Kommentar
  `voxel_buffer.h:165-166` von 16 Bit erhöht, dadurch 256 Materialien statt 16
- `CHANNEL_WEIGHTS` 16 Bit, 4 Bit je Slot (`voxel_buffer.h:166`)
- `CHANNEL_DATA5` dient dem Fluid-Isosurface-Pass
  (`voxel_mesher_hybrid.h:79-80`, `voxel_mesher_hybrid.cpp:571-575`)

Die Speicheranordnung ist ZXY, belegt durch den Kommentar und die
Sprungweiten in `voxel_mesher_hybrid.cpp:753-761` mit `y_jump = 1`,
`x_jump = size.y`, `z_jump = size.y * size.x`.

### 1.5 Build-Reihenfolge in `build()`

`voxel_mesher_hybrid.cpp:524-581` legt eine feste Reihenfolge fest, deren
Begründungen in den Kommentaren stehen und im Port erhalten bleiben müssen:

1. Optionales Abflachen des SDF unter Modellen, danach Transvoxel-Terrain-Pass
   (`:529-556`). Der abgeflachte Puffer ist `thread_local` und bewusst mit
   `ALLOCATOR_DEFAULT` statt Pool angelegt (`:531-534`).
2. `finalize_terrain_surfaces` setzt den Material-ID-Stream, muss vor dem
   Fluid-Pass laufen (`:558-563`).
3. `promote_terrain_collision` wandelt den Submesh-Bereich in explizite
   Kollisionsarrays, muss vor dem Fluid-Pass laufen (`:565-569`).
4. `build_fluid_surfaces` zweiter Transvoxel-Pass auf `CHANNEL_DATA5`, ohne
   Kollisionsbeitrag (`:571-575`).
5. `build_model_surface` Blocky-Modelle, Material-ID N wird
   Surface-Index 2 + N (`:577-580`).

`build_fluid_surfaces` sichert vor dem inneren Aufruf `submesh_vertex_end`,
`submesh_index_end`, `mesh_flags` und die Surface-Anzahl und stellt sie danach
wieder her (`:599-610`). Grund laut Kommentar, der innere `build()` setzt
`mesh_flags` mit Zuweisung statt Veroderung.

Die Material-Konvention ist 0 Terrain, 1 Wasser, ab 2 Bibliotheksmaterialien
(`voxel_mesher_hybrid.h:41-46`, `WATER_MATERIAL_INDEX = 1` in `:53`).

Die Ablage der Material-ID hängt vom Texturierungsmodus ab. Bei
`TEXTURES_NONE` liegt sie in CUSTOM1, bei aktiver Texturierung belegt
Transvoxel CUSTOM1 mit zwei Floats und die ID wandert nach CUSTOM2
(`voxel_mesher_hybrid.h:105-112`, `voxel_mesher_hybrid.cpp:617-640`).

### 1.6 Blocky-Modellpfad

`voxel_mesher_hybrid.cpp:727-846` zeigt die benötigten Blocky-Bausteine:

- `blocky::AIR_ID` als Leerkennung, Frühabbruch bei uniformem Block (`:738-741`)
- `MODEL_PADDING = 1` (`:28`), Randstreifen wird übersprungen (`:744-749`)
- `_library->get_baked_data_rw_lock()` und `blocky::BakedLibrary`, nur die
  Backdaten sind threadsicher (`:768-770`)
- `blocky::is_face_visible` für Seitenkulling über den Nachbarvoxel (`:812-816`)
- `model.empty_sides_mask`, Fluidmodelle markieren SIDE_POSITIVE_Y als leer
  (`:802-806`, `:808-811`)
- `blocky::generate_fluid_model` mit Interpolation der Eckhöhen aus neun
  horizontalen Nachbarn, benötigt Rohzugriff auf den TYPE-Puffer (`:828-846`)

Die Backdatenstrukturen liegen in `meshers/blocky/blocky_baked_library.h`,
`BakedModel::SideSurface` und `BakedModel::Surface` mit Positionen, UVs,
Indizes, Tangenten und `material_id` (dort Zeilen 40-78). Grenzen sind
`MAX_MODELS = 65536`, `MAX_FLUIDS = 256`, `MAX_MATERIALS = 65536`,
`MAX_SURFACES = 2`, `AIR_ID = 0`, `NULL_FLUID_INDEX = 255` (ebenda 22-35).

---

## 2. Lage im Rust- und Bevy-Ökosystem

Bevy 0.18 erstellt Meshes über `Mesh::new` mit `PrimitiveTopology`, Indizes
über `with_inserted_indices`, Attribute über `insert_attribute` oder
`with_inserted_attribute`. Für eigene Vertexströme gilt
`FIRST_AVAILABLE_CUSTOM_ATTRIBUTE` mit Wert 8, darunterliegende Indizes sind
Bevy-eigen. Quellen siehe Abschnitt 6.

Damit lassen sich die hier benötigten CUSTOM0 bis CUSTOM2 als eigene
Attribute abbilden, allerdings nicht über Godots ARRAY_CUSTOM-Slots, sondern
über frei gewählte Attribut-IDs ab 8 mit passendem `VertexFormat`.

Der Crate `transvoxel` liegt in Version 2.0.0 vor, Stand 11.11.2025, lizenziert
Apache-2.0 oder MIT, mit `DataField`-Trait als Dichtequelle, `VoxelVecBlock`
als Caching und Transition-Sides für LOD-Übergänge. Materialien oder
Texturkoordinaten liefert der Crate nach eigener Dokumentation nicht.

Konsequenz. Als Geometriegerüst ist der Crate nutzbar, der hier benötigte
Mixel4-Materialpfad und die 4-Slot-Gewichtsinterpolation sind darin nicht
enthalten und müssen in jedem Fall aus
`meshers/transvoxel/transvoxel_materials_mixel4.h` (326 Zeilen) und
`storage/mixel4.h` (209 Zeilen) portiert werden.

Lizenzlage. Dieses Repository steht unter MIT
(`LICENSE.md`, Copyright 2016-2024 Marc Gilleron). Eine Portierung und
Weiterverwendung ist damit zulässig, Copyright- und Lizenzhinweis müssen
mitgeführt werden.

---

## 3. Architekturentscheidung

Vorgeschlagen wird eine Trennung in drei Crates innerhalb eines Workspaces.

```
voxel-core      kein Bevy, kein Godot. VoxelBuffer, Kanäle, Tiefen,
                Mixel4, Box3i, Vector3f-Äquivalente, cube_tables.
voxel-mesher    kein Bevy. Transvoxel, Blocky-Backdaten, Hybrid-Logik.
                Ausgabe als engine-neutrale MeshOutput-Struktur.
bevy-voxel      Bevy-Plugin. MeshOutput zu bevy_mesh::Mesh, Material,
                Chunk-Entities, Tasks, SDF-Edit-API, Raycast.
```

Begründung. Der Transvoxel-Kern ist laut Abschnitt 1.3 kaum an Godot gebunden,
die Mesher-Basis dagegen stark. Eine neutrale `MeshOutput`-Struktur erlaubt
Unit-Tests ohne Renderer und hält die Bevy-Version aus dem Algorithmuskern
heraus.

Vorgeschlagene neutrale Ausgabestruktur, Ersatz für
`VoxelMesher::Output` (`meshers/voxel_mesher.h:50-77`):

```rust
pub struct Surface {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Option<Vec<Vec2>>,
    pub indices: Vec<u32>,
    // CUSTOM0 of transvoxel, cell info
    pub custom0: Option<Vec<u32>>,
    // texture data, 2 floats, only when texturing is enabled
    pub custom1: Option<Vec<[f32; 2]>>,
    // per vertex material id
    pub material_id_stream: Vec<f32>,
    pub material_index: u16,
}

pub struct MeshOutput {
    pub surfaces: Vec<Surface>,
    pub transition_surfaces: [Vec<Surface>; 6],
    pub collision: CollisionSurface,
}
```

Abweichung zur C++-Fassung. Dort wird die Material-ID je nach
Texturierungsmodus in CUSTOM1 oder CUSTOM2 geschrieben
(`voxel_mesher_hybrid.cpp:617-640`). In Rust wird sie als eigenes Feld geführt
und erst beim Bevy-Upload auf eine Attribut-ID gelegt. Damit entfällt die
Fehlerquelle widersprüchlicher Vertexformate zwischen Surfaces, die im
C++-Code mehrfach kommentiert ist (`:612-616`).

---

## 4. Arbeitspakete

Die Reihenfolge ist so gewählt, dass jedes Paket für sich testbar ist.

### P0 Vorbereitung

- Cargo-Workspace mit den drei Crates anlegen, Edition 2021, Bevy 0.18 nur in
  `bevy-voxel`.
- Rust-Toolchain bereitstellen. Im aktuellen Container ist keine vorhanden,
  `which cargo rustc` liefert nichts.
- MIT-Hinweis aus `LICENSE.md` in die neuen Crates übernehmen.
- Testkorpus erzeugen. Dazu im C++-Modul mehrere `VoxelBuffer` mit SDF-,
  TYPE-, INDICES-, WEIGHTS- und DATA5-Inhalt serialisieren und als
  Referenzdaten ablegen. Grundlage dafür sind die bestehenden Tests unter
  `tests/voxel/`.

Ergebnis. Leerer, baubarer Workspace plus Referenzdaten.

### P1 voxel-core, Datenhaltung

Portierung aus `storage/voxel_buffer.{h,cpp}`, 2354 Zeilen, davon im
Hybrid-Pfad benötigt:

- Kanalenum und Tiefen (`voxel_buffer.h:40-68`)
- `real_to_raw_voxel` und `raw_voxel_to_real` einschliesslich der
  Normalisierung unter 32 Bit (`voxel_buffer.h:106-158`)
- Standardwerte je Kanal (`voxel_buffer.h:161-221`)
- ZXY-Indizierung, uniforme Kompression, `get_channel_as_bytes_read_only`
- `storage/mixel4.h` vollständig, 209 Zeilen
- `constants/cube_tables.*`, 346 Zeilen, insbesondere `g_side_normals` und
  `SIDE_COUNT`, benutzt in `voxel_mesher_hybrid.cpp:808-816`

Nicht portiert werden Serialisierung, Metadaten
(`storage/metadata/`), Speicherpool (`voxel_memory_pool.*`), `voxel_data*`
und `voxel_buffer_gd.*`. Begründung, diese dienen Streaming, Persistenz und
Godot-Bindung und liegen ausserhalb des Auftrags.

Test. Byteweise Gleichheit der Kanalinhalte gegen die Referenzdaten aus P0,
sowie Rundlauf `real_to_raw` zu `raw_to_real` je Tiefe.

Aufwand, geschätzt 1200 bis 1500 Zeilen Rust.

### P2 voxel-mesher, Transvoxel-Geometrie

Portierung aus `meshers/transvoxel/transvoxel.{h,cpp}` und
`transvoxel_tables.cpp`, zusammen 2838 Zeilen.

- Tabellen maschinell übertragen. `transvoxel_tables.cpp` ist reine Daten,
  ein Konvertierungsskript vermeidet Übertragungsfehler.
- Zellen-Iteration, Vertex-Reuse, Normalen aus SDF-Gradienten.
- Transition-Flächen je Seite, Ablage in `transition_surfaces`.

Alternative. Statt Eigenportierung den Crate `transvoxel` 2.0.0 als
Geometriequelle verwenden. Vorteil, 2838 Zeilen entfallen. Nachteil, der
Materialpfad fehlt dort und die Vertex-Reuse-Indizierung ist nicht
kompatibel mit dem Mixel4-Pfad, der je Vertex Gewichte interpoliert.
Empfehlung, Eigenportierung, weil P3 sonst gegen eine fremde interne
Vertexstruktur arbeiten muss.

Test. Vertex- und Indexanzahl sowie Positions-Hashes gegen die vom
C++-Mesher auf denselben Referenzpuffern erzeugten Werte.

Aufwand, geschätzt 2500 bis 3000 Zeilen Rust.

### P3 voxel-mesher, Transvoxel-Materialien

Portierung aus `transvoxel_materials_mixel4.h` (326),
`transvoxel_materials_single_s4.h` (249),
`transvoxel_materials_single_s2.h` (229),
`transvoxel_materials_single_common.h` (127),
`transvoxel_materials_common.h` (45), `transvoxel_materials_null.h` (29),
zusammen 1005 Zeilen.

Mindestumfang für `rame`. Der Generator dort schreibt Mixel4-Materialien
(`rame/godot-voxel-cpp/terrain_generator/generator.h`, Klassenkommentar), also
sind `mixel4` und `null` zwingend. Die `single_s2`- und `single_s4`-Varianten
sind optional und können zurückgestellt werden.

Zu beachten ist die Option `textures_ignore_air_voxels`
(`voxel_mesher_hybrid.h:113-118`). Laut Kommentar verhindert sie, dass
Generator-Standardindizes auf Luftvoxeln in Edits einbluten.

Test. Vergleich der CUSTOM1-Texturdaten gegen C++-Referenz, je Vertex.

Aufwand, geschätzt 700 bis 900 Zeilen Rust.

### P4 voxel-mesher, Blocky-Backdaten und Modellpfad

Portierung aus `blocky_baked_library.h` (201), Teile von
`voxel_mesher_blocky.cpp` (952) und `blocky_fluids_meshing_impl.h` (440).

Benötigt wird:

- `BakedModel`, `BakedLibrary`, `Surface`, `SideSurface`, Grenzen und
  `AIR_ID`, `NULL_FLUID_INDEX`
- `is_face_visible` einschliesslich Transparenz- und Seitenmuster-Test
- `generate_fluid_model` mit den Neun-Nachbar-Höhen
- `empty_sides_mask`-Behandlung

Nicht portiert werden die Godot-Ressourcenklassen
`voxel_blocky_model*.cpp`, `voxel_blocky_library*.cpp`,
`voxel_blocky_fluid.cpp`, zusammen rund 2900 Zeilen. Begründung, diese
bilden den Editor- und Ressourcenpfad, der die Backdaten erst erzeugt. In
Bevy tritt an deren Stelle ein eigenes Asset-Format.

Stattdessen wird ein schlanker Bibliotheks-Loader benötigt, der Backdaten
aus einer Rust-seitigen Beschreibung erzeugt. Vorschlag, RON oder JSON mit
Modellliste, Materialzuordnung, Transparenzflag, Fluid-Index, plus Laden
externer Meshes über `bevy_asset`. Das ist Neuentwicklung, nicht Portierung,
und mit etwa 400 bis 600 Zeilen anzusetzen.

Test. Für einen Referenzblock mit gesetzten TYPE-Werten Vergleich der
Dreiecksanzahl je Material-ID gegen C++.

Aufwand, geschätzt 1200 bis 1600 Zeilen Rust plus Loader.

### P5 voxel-mesher, Hybrid-Logik

Portierung von `voxel_mesher_hybrid.cpp`, 1083 Zeilen, ohne `_bind_methods`
(`:984-1083`, rund 100 Zeilen entfallen).

Reihenfolge und Zustandssicherung strikt wie in Abschnitt 1.5. Teilschritte:

- `build_flattened_sdf` (`:340-523`, 184 Zeilen), Abflachen des SDF auf die
  Unterkante von Modellvoxeln
- `build` (`:524-581`)
- `build_fluid_surfaces` (`:583-643`)
- `finalize_terrain_surfaces` (`:645-685`)
- `promote_terrain_collision` (`:686-726`)
- `build_model_surface` (`:727-983`, 257 Zeilen)

In Rust entfällt das `thread_local`-Problem aus `:531-534`, weil der
abgeflachte Puffer als lokaler, von einem Pool geborgter Wert geführt werden
kann. Der Kommentargrund, Pool-Blöcke überleben den Aufruf, trifft dann nicht
mehr zu.

Ebenso entfällt die Zustandssicherung aus `:599-610`, sobald der Fluid-Pass
in einen eigenen `MeshOutput` schreibt, der danach zusammengeführt wird.
Dieser Unterschied ist bewusst und sollte im Code kommentiert werden.

Test. Vollständiger Vergleich aller Surfaces, Material-Indizes und der
Kollisionsarrays gegen C++ für mindestens fünf Referenzblöcke, darunter einer
mit Fluid, einer mit Modellen über Chunkgrenze und einer uniformer Luftblock.

Aufwand, geschätzt 1100 bis 1400 Zeilen Rust.

### P6 bevy-voxel, Mesh-Erzeugung

- `MeshOutput` nach `bevy_mesh::Mesh`, `PrimitiveTopology::TriangleList`,
  Attribute Position, Normal, optional UV, Indizes als `Indices::U32`.
- Eigene Attribute ab `FIRST_AVAILABLE_CUSTOM_ATTRIBUTE` gleich 8 für CUSTOM0,
  Texturdaten und Material-ID.
- Ersatz für `get_default_lod_material` (`voxel_mesher_hybrid.h:68`). Der
  C++-Shader liegt in `shaders/voxel_mesher_hybrid_shader.h`, 51 Zeilen, und
  leistet laut Kommentar (`voxel_mesher_hybrid.h:58-67`) zwei Dinge,
  Culling der Transition-Dreiecke und Farbwahl je Vertex aus der Material-ID.
  In Bevy wird dafür ein WGSL-Material mit `AsBindGroup` benötigt, inklusive
  `u_material_colors`-Äquivalent, gespeist aus `terrain_color` und
  `model_colors` (`voxel_mesher_hybrid.h:94-98`).
- Aufteilung Terrain- und Wassermesh anhand `material_index == 1`, analog der
  in `voxel_mesher_hybrid.h:48-56` beschriebenen Aufteilung durch
  `VoxelLodTerrain`.

Aufwand, geschätzt 600 bis 900 Zeilen Rust plus WGSL.

### P7 bevy-voxel, Chunkverwaltung und Nebenläufigkeit

Dieses Paket hat im C++-Repo kein direkt portierbares Gegenstück im erlaubten
Umfang. `terrain/` umfasst 25540 Zeilen und `engine/` 5330 Zeilen und ist
ausdrücklich nicht Teil des Auftrags.

Benötigter Mindestumfang, damit der Mesher nutzbar ist:

- Chunk-Entities mit Position, LOD-Index, Zustand
- Pufferbereitstellung mit Randstreifen. Padding liefert
  `get_minimum_padding` und `get_maximum_padding`
  (`meshers/voxel_mesher.h:91-95`), Modellpadding ist 1
  (`voxel_mesher_hybrid.cpp:28`).
- Meshing auf `AsyncComputeTaskPool`, da `build` laut
  `meshers/voxel_mesher.h:81` mehrfach parallel aufgerufen werden darf
- LOD-Auswahl und Transition-Masken je Chunkseite

Bewertung. Dies ist der grösste Neuentwicklungsanteil. Als Abkürzung kommt
`bevy_voxel_world` in Frage, das laut Suchergebnis LOD je Chunk über
`VoxelWorldConfig::chunk_lod` erlaubt und gepaddete Dimensionen je LOD
definiert. Ob dessen Mesher-Schnittstelle einen Fremdmesher aufnehmen kann,
ist nicht verifiziert und muss vor einer Festlegung geprüft werden.

Aufwand, geschätzt 1500 bis 2500 Zeilen Rust bei Eigenentwicklung.

### P8 bevy-voxel, Edit- und Raycast-API

Ziel ist Funktionsgleichheit mit der in Abschnitt 1.2 ermittelten Oberfläche,
die `rame` tatsächlich nutzt.

- `carve_sdf_sphere`, `fill_sdf_sphere`, `place_sdf_box`, `carve_sdf_voxel`,
  `raycast`
- Der SDF-Kugelpfad wächst laut Kommentar in
  `rame/rust/game/src/config/voxel_edits_config.rs` die Arbeitsbox um zwei
  Voxel Rand, Verweis dort auf `edition/funcs.h:316-323`. Dieses Verhalten ist
  bei der Portierung zu prüfen und zu übernehmen.
- `do_box` leitet die halbe Grösse aus `(end - begin) * 0.5` ab, Verweis in
  derselben `rame`-Quelle auf `voxel_tool_lod_terrain.cpp:108-109`.
- `do_point` schreibt in `MODE_REMOVE` den Wert `SDF_FAR_OUTSIDE`, Verweis auf
  `voxel_tool.cpp:153-155`.

Quelle im C++-Repo ist `edition/funcs.{h,cpp}` und `edition/voxel_tool*.cpp`.
Nur die SDF-Operationen Kugel, Box, Punkt und der Raycast werden portiert,
nicht `mesh_sdf`, `floating_chunks` oder die Buffer- und
Terrain-Tool-Varianten.

Aufwand, geschätzt 500 bis 800 Zeilen Rust.

### P9 Generator-Schnittstelle

Der `rame`-Generator ist C++ und erbt von `VoxelGeneratorHeightmap`. Im Port
wird daraus ein Rust-Trait.

Minimale Entsprechung zu `generators/voxel_generator.h:63-69`:

```rust
pub trait VoxelGenerator: Send + Sync {
    fn generate_block(&self, buffer: &mut VoxelBuffer, origin: IVec3, lod: u32)
        -> GenerateResult;
    fn used_channels_mask(&self) -> u32;
}
```

`GenerateResult` trägt das `max_lod_hint` aus
`generators/voxel_generator.h:56-60`.

Die Biome-, Höhlen- und Plot-Logik aus `rame/godot-voxel-cpp` ist Teil des
Fremdprojekts und nicht Gegenstand dieses Plans. Die dort genutzten
Rauschquellen `ZN_FastNoiseLite` und `FastNoise2` haben in Rust
Entsprechungen, deren Auswahl offen bleibt und vor P9 zu klären ist.

Aufwand für Trait und einen Referenzgenerator, geschätzt 200 bis 400 Zeilen.

---

## 5. Umfang, Risiken, offene Punkte

### 5.1 Umfangsbilanz

Zu portierender C++-Code im Mindestumfang, ohne Editor, Streams, Terrain,
Engine, Modifiers und Graph-Generatoren:

| Paket | C++-Zeilen Quelle | Rust, geschätzt |
|---|---|---|
| P1 Storage und Konstanten | rund 2900 | 1200 bis 1500 |
| P2 Transvoxel-Geometrie | 2838 | 2500 bis 3000 |
| P3 Transvoxel-Materialien | 1005 | 700 bis 900 |
| P4 Blocky-Backdaten | rund 1600 | 1200 bis 1600 |
| P5 Hybrid-Logik | 983 | 1100 bis 1400 |
| P6 Bevy-Mesh und Material | 51 plus neu | 600 bis 900 |
| P7 Chunkverwaltung | Neuentwicklung | 1500 bis 2500 |
| P8 Edits und Raycast | Teile von `edition/` | 500 bis 800 |
| P9 Generator-Trait | rund 100 | 200 bis 400 |

Summe, grob 9500 bis 13000 Zeilen Rust. Der Anteil echter Neuentwicklung
liegt bei P4-Loader, P6-Material, P7 und damit bei rund einem Drittel.

### 5.2 Risiken

- P7 ist der kritische Pfad. Ohne Chunkverwaltung und LOD-Logik ist der
  Mesher nicht sichtbar lauffähig, und dieser Teil ist im Auftrag nicht
  enthalten, aber faktisch erforderlich.
- Die Transvoxel-Tabellen müssen fehlerfrei übertragen werden. Manuelle
  Übertragung ist auszuschliessen.
- Der Mixel4-Pfad hat kein Gegenstück im `transvoxel`-Crate, siehe
  Abschnitt 2. Eine Abkürzung über den Crate verschiebt den Aufwand nur.
- Das Vertexformat muss über alle Surfaces eines Chunks einheitlich sein.
  Im C++-Code ist dies mehrfach als Fehlerquelle dokumentiert
  (`voxel_mesher_hybrid.cpp:592-598`, `:612-616`). In Bevy ist die
  Fehlermeldung anders, die Ursache dieselbe.
- Die 256-Material-Erweiterung über 32-Bit-INDICES
  (`voxel_buffer.h:165-166`, Commit `be210a94`) muss im Port erhalten bleiben,
  sonst bricht der `rame`-Generator.

### 5.3 Vor Beginn zu klärende Punkte

- Soll das Bevy-Ziel `rame` ersetzen oder parallel bestehen. Die Antwort
  entscheidet, ob P8 die gdext-Signaturen nachbildet oder frei gestaltet wird.
- Zielversion von Bevy. Hier wurde gegen 0.18 recherchiert.
- Rauschbibliothek für P9.
- Eigener Chunkverwalter oder `bevy_voxel_world` als Grundlage für P7.
- Ob die Varianten `single_s2` und `single_s4` aus P3 überhaupt benötigt
  werden.

---

## 6. Quellen

Repository, geprüft am 2026-11-10 auf Branch `hybrid-mesher`, Commit
`08947db0`:

- `meshers/hybrid/voxel_mesher_hybrid.h`, `meshers/hybrid/voxel_mesher_hybrid.cpp`
- `meshers/voxel_mesher.h`
- `meshers/transvoxel/`, `meshers/blocky/`
- `storage/voxel_buffer.h`, `storage/mixel4.h`
- `constants/cube_tables.h`
- `generators/voxel_generator.h`
- `LICENSE.md`

Fremdprojekt `/home/ubuntu/git/rame`, geprüft am 2026-11-10:

- `rust/Cargo.toml`, `rust/game/Cargo.toml`, `rust/scene_nodes/Cargo.toml`
- `rust/.cargo/config.toml`
- `rust/game/src/lib.rs`, `rust/game/src/voxel_edits/voxel_edits.rs`
- `rust/game/src/config/world_config.rs`, `rust/game/src/config/voxel_edits_config.rs`
- `rust/game/src/scenes/world_nodes.rs`
- `godot/rame.gdextension`, `godot/scenes/world.tscn`
- `godot-voxel-cpp/terrain_generator/generator.h`

Extern:

- [bevy_render 0.18.0 mesh module](https://docs.rs/crate/bevy_render/0.18.0/source/src/mesh/mod.rs)
- [Bevy Beispiel, Generate Custom Mesh](https://bevy.org/examples/3d-rendering/generate-custom-mesh/)
- [Struct Mesh, bevy latest](https://docs.rs/bevy/latest/bevy/mesh/struct.Mesh.html)
- [transvoxel 2.0.0 README](https://docs.rs/crate/transvoxel/2.0.0/source/README.md)
- [Crate transvoxel 1.0.0](https://docs.rs/crate/transvoxel/1.0.0)
- [bevy_voxel_world](https://rust-digger.code-maven.com/crates/bevy_voxel_world)
- [bevy_meshem](https://rust-digger.code-maven.com/crates/bevy_meshem)
- [Smooth terrains, voxel-tools Dokumentation](https://voxel-tools.readthedocs.io/en/latest/smooth_terrain/)
