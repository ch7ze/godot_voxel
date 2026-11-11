"""Convert transvoxel_tables.cpp lookup tables into a Rust module.

Mechanical transcription only, no reformatting of values. Run:
  python3 -I convert_transvoxel_tables.py <in.cpp> <out.rs>
"""
import re
import sys


def read_source(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def extract_block(text, decl):
    """Returns the text between the '{' following decl and its matching '}'."""
    start = text.index(decl)
    brace = text.index("{", start + len(decl))
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i]
    raise ValueError(f"unbalanced braces after {decl!r}")


def parse_flat(block):
    return [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+|\d+", block)]


def parse_rows(block, expected_rows):
    """Splits a brace-nested initializer into its top-level rows."""
    rows = []
    depth = 0
    current = []
    for ch in block:
        if ch == "{":
            depth += 1
            if depth == 1:
                current = []
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rows.append(parse_flat("".join(current)))
                continue
        if depth >= 1:
            current.append(ch)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, parsed {len(rows)}")
    return rows


def fmt_u8_array(values, per_line=16):
    out = []
    for i in range(0, len(values), per_line):
        chunk = ", ".join(f"0x{v:02x}" for v in values[i : i + per_line])
        out.append("    " + chunk + ",")
    return "\n".join(out)


def fmt_padded_rows(rows, width, fill, hex_digits):
    out = []
    for row in rows:
        if len(row) > width:
            raise ValueError(f"row longer than {width}: {len(row)}")
        padded = list(row) + [fill] * (width - len(row))
        body = ", ".join(f"0x{v:0{hex_digits}x}" for v in padded)
        out.append(f"    [{body}],")
    return "\n".join(out)


def fmt_cell_data(rows, index_width):
    out = []
    for row in rows:
        counts = row[0]
        indices = row[1:]
        if len(indices) > index_width:
            raise ValueError(f"too many indices: {len(indices)}")
        padded = indices + [0] * (index_width - len(indices))
        body = ", ".join(str(v) for v in padded)
        out.append(
            f"    CellData {{ geometry_counts: 0x{counts:02x}, "
            f"vertex_index: [{body}] }},"
        )
    return "\n".join(out)


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    text = strip_comments(read_source(src_path))

    regular_cell_class = parse_flat(
        extract_block(text, "const unsigned char regularCellClass[256]")
    )
    assert len(regular_cell_class) == 256, len(regular_cell_class)

    regular_cell_data = parse_rows(
        extract_block(text, "const RegularCellData regularCellData[16]"), 16
    )

    regular_vertex_data = parse_rows(
        extract_block(text, "const unsigned short regularVertexData[256][12]"), 256
    )

    transition_cell_class = parse_flat(
        extract_block(text, "const unsigned char transitionCellClass[512]")
    )
    assert len(transition_cell_class) == 512, len(transition_cell_class)

    transition_cell_data = parse_rows(
        extract_block(text, "const TransitionCellData transitionCellData[56]"), 56
    )

    transition_corner_data = parse_flat(
        extract_block(text, "const unsigned char transitionCornerData[13]")
    )
    assert len(transition_corner_data) == 13, len(transition_corner_data)

    transition_vertex_data = parse_rows(
        extract_block(text, "const unsigned short transitionVertexData[512][12]"), 512
    )

    parts = []
    parts.append(
        """//================================================================================
//
// The Transvoxel Algorithm look-up tables
//
// Copyright 2009 by Eric Lengyel
//
// The following data originates from Eric Lengyel's Transvoxel Algorithm.
// http://transvoxel.org/
//
// The data in this file may be freely used in implementations of the Transvoxel
// Algorithm. If you do use this data, or any transformation of it, in your own
// projects, commercial or otherwise, please give credit by indicating in your
// source code that the data is part of the author's implementation of the
// Transvoxel Algorithm and that it came from the web address given above.
//
//================================================================================
//
// GENERATED FILE. Transcribed mechanically from
// `meshers/transvoxel/transvoxel_tables.cpp` of the C++ module by
// `tools/convert_transvoxel_tables.py`. Do not edit by hand.

/// Triangulation data of one equivalence class.
/// `geometry_counts` packs the vertex count in the high nibble and the
/// triangle count in the low nibble.
/// Source: meshers/transvoxel/transvoxel_tables.cpp:41-83
#[derive(Debug, Clone, Copy)]
pub struct CellData<const N: usize> {
    pub geometry_counts: u8,
    pub vertex_index: [u8; N],
}

impl<const N: usize> CellData<N> {
    pub const fn vertex_count(&self) -> usize {
        (self.geometry_counts >> 4) as usize
    }

    pub const fn triangle_count(&self) -> usize {
        (self.geometry_counts & 0x0f) as usize
    }

    pub fn get_vertex_index(&self, i: usize) -> u8 {
        self.vertex_index[i]
    }
}

/// Triangulation of a regular cell, at most 5 triangles.
pub type RegularCellData = CellData<15>;
/// Triangulation of a transition cell, at most 12 triangles.
pub type TransitionCellData = CellData<36>;
"""
    )

    parts.append(
        "/// Maps an 8-bit regular Marching Cubes case index to an equivalence class.\n"
        "/// Source: meshers/transvoxel/transvoxel_tables.cpp:91-108\n"
        "pub const REGULAR_CELL_CLASS: [u8; 256] = [\n"
        + fmt_u8_array(regular_cell_class)
        + "\n];\n"
    )

    parts.append(
        "/// Triangulation data of the 16 distinct regular classes.\n"
        "/// Unused index slots are zero-filled, read only up to\n"
        "/// `triangle_count() * 3`.\n"
        "/// Source: meshers/transvoxel/transvoxel_tables.cpp:120-137\n"
        "pub const REGULAR_CELL_DATA: [RegularCellData; 16] = [\n"
        + fmt_cell_data(regular_cell_data, 15)
        + "\n];\n"
    )

    parts.append(
        "/// Vertex locations and reuse data for all 256 regular cases.\n"
        "/// Low byte holds the two edge endpoints, high byte the reuse data.\n"
        "/// Rows shorter than 12 are zero-filled, read only up to\n"
        "/// `vertex_count()`.\n"
        "/// Source: meshers/transvoxel/transvoxel_tables.cpp:152-409\n"
        "pub const REGULAR_VERTEX_DATA: [[u16; 12]; 256] = [\n"
        + fmt_padded_rows(regular_vertex_data, 12, 0, 4)
        + "\n];\n"
    )

    parts.append(
        "/// Maps a 9-bit transition cell case index to an equivalence class.\n"
        "/// Source: meshers/transvoxel/transvoxel_tables.cpp:426-459\n"
        "pub const TRANSITION_CELL_CLASS: [u8; 512] = [\n"
        + fmt_u8_array(transition_cell_class)
        + "\n];\n"
    )

    parts.append(
        "/// Triangulation data of the 56 distinct transition classes.\n"
        "/// Source: meshers/transvoxel/transvoxel_tables.cpp:472-529\n"
        "pub const TRANSITION_CELL_DATA: [TransitionCellData; 56] = [\n"
        + fmt_cell_data(transition_cell_data, 36)
        + "\n];\n"
    )

    parts.append(
        "/// Sample indices of the 13 transition cell corners.\n"
        "/// Source: meshers/transvoxel/transvoxel_tables.cpp:541-543\n"
        "pub const TRANSITION_CORNER_DATA: [u8; 13] = [\n"
        + fmt_u8_array(transition_corner_data)
        + "\n];\n"
    )

    parts.append(
        "/// Vertex locations and reuse data for all 512 transition cases.\n"
        "/// Source: meshers/transvoxel/transvoxel_tables.cpp:558-1071\n"
        "pub const TRANSITION_VERTEX_DATA: [[u16; 12]; 512] = [\n"
        + fmt_padded_rows(transition_vertex_data, 12, 0, 4)
        + "\n];\n"
    )

    parts.append(
        """#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_cell_class_indexes_existing_data() {
        for case in 0..256 {
            assert!((REGULAR_CELL_CLASS[case] as usize) < REGULAR_CELL_DATA.len());
        }
    }

    #[test]
    fn transition_cell_class_indexes_existing_data() {
        // Bit 7 of the class carries the triangle flip flag and must be
        // masked off before indexing.
        // Source: meshers/transvoxel/transvoxel_tables.cpp:468-469
        for case in 0..512 {
            let class = (TRANSITION_CELL_CLASS[case] & 0x7f) as usize;
            assert!(class < TRANSITION_CELL_DATA.len(), "case {case}");
        }
    }

    #[test]
    fn regular_vertex_counts_match_the_class_triangulation() {
        for case in 1..255 {
            let data = &REGULAR_CELL_DATA[REGULAR_CELL_CLASS[case] as usize];
            let vertex_count = data.vertex_count();
            let non_zero = REGULAR_VERTEX_DATA[case].iter().filter(|v| **v != 0).count();
            assert_eq!(non_zero, vertex_count, "case {case}");
            // Every triangulation index addresses an existing vertex
            for i in 0..data.triangle_count() * 3 {
                assert!(
                    (data.get_vertex_index(i) as usize) < vertex_count,
                    "case {case} index {i}"
                );
            }
        }
    }

    #[test]
    fn transition_vertex_counts_match_the_class_triangulation() {
        for case in 0..512 {
            let data = &TRANSITION_CELL_DATA[TRANSITION_CELL_CLASS[case] as usize & 0x7f];
            let vertex_count = data.vertex_count();
            let non_zero = TRANSITION_VERTEX_DATA[case].iter().filter(|v| **v != 0).count();
            assert_eq!(non_zero, vertex_count, "case {case}");
            for i in 0..data.triangle_count() * 3 {
                assert!(
                    (data.get_vertex_index(i) as usize) < vertex_count,
                    "case {case} index {i}"
                );
            }
        }
    }

    #[test]
    fn empty_and_full_regular_cases_produce_no_geometry() {
        assert_eq!(REGULAR_CELL_CLASS[0], 0);
        assert_eq!(REGULAR_CELL_CLASS[255], 0);
        assert_eq!(REGULAR_CELL_DATA[0].vertex_count(), 0);
        assert_eq!(REGULAR_CELL_DATA[0].triangle_count(), 0);
    }

    #[test]
    fn regular_triangle_counts_stay_within_the_documented_maximum() {
        // Source: meshers/transvoxel/transvoxel.h:20, MAX_TRIANGLES_PER_CELL
        for data in REGULAR_CELL_DATA {
            assert!(data.triangle_count() <= 5);
        }
    }
}
"""
    )

    with open(dst_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))

    print(f"wrote {dst_path}")
    print(f"regular_cell_class {len(regular_cell_class)}")
    print(f"regular_cell_data {len(regular_cell_data)}")
    print(f"regular_vertex_data {len(regular_vertex_data)}")
    print(f"transition_cell_class {len(transition_cell_class)}")
    print(f"transition_cell_data {len(transition_cell_data)}")
    print(f"transition_corner_data {len(transition_corner_data)}")
    print(f"transition_vertex_data {len(transition_vertex_data)}")


main()
