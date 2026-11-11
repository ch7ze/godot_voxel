//! Integer box helper ported from `util/math/box3i.h`.

use glam::IVec3;

/// Axis-aligned integer box, stored as position plus size.
/// Source: util/math/box3i.h:16-20
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Box3i {
    pub position: IVec3,
    pub size: IVec3,
}

/// Clamps a 1D range into a limit range, collapsing to zero size on no overlap.
/// Source: util/math/box3i.h:301-312
fn clip_range(pos: &mut i32, size: &mut i32, lim_pos: i32, lim_size: i32) {
    let mut max_pos = *pos + *size;
    let lim_max_pos = lim_pos + lim_size;

    *pos = (*pos).clamp(lim_pos, lim_max_pos);
    max_pos = max_pos.clamp(lim_pos, lim_max_pos);

    *size = max_pos - *pos;
    if *size < 0 {
        *size = 0;
    }
}

impl Box3i {
    pub const fn new(position: IVec3, size: IVec3) -> Box3i {
        Box3i { position, size }
    }

    /// Source: util/math/box3i.h:25-27
    pub fn from_center_extents(center: IVec3, extents: IVec3) -> Box3i {
        Box3i::new(center - extents, 2 * extents)
    }

    /// Source: util/math/box3i.h:30-32
    pub fn from_min_max(min: IVec3, max: IVec3) -> Box3i {
        Box3i::new(min, max - min)
    }

    /// Box covering a volume of the given size at the origin.
    pub fn from_size(size: IVec3) -> Box3i {
        Box3i::new(IVec3::ZERO, size)
    }

    pub fn end(&self) -> IVec3 {
        self.position + self.size
    }

    pub fn volume(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            self.size.x as u64 * self.size.y as u64 * self.size.z as u64
        }
    }

    /// Upper bound is exclusive.
    /// Source: util/math/box3i.h:51-59
    pub fn contains_point(&self, pos: IVec3) -> bool {
        let end = self.end();
        pos.x >= self.position.x
            && pos.y >= self.position.y
            && pos.z >= self.position.z
            && pos.x < end.x
            && pos.y < end.y
            && pos.z < end.z
    }

    /// Source: util/math/box3i.h:61-70
    pub fn contains_box(&self, other: Box3i) -> bool {
        let other_end = other.end();
        let end = self.end();
        other.position.x >= self.position.x
            && other.position.y >= self.position.y
            && other.position.z >= self.position.z
            && other_end.x <= end.x
            && other_end.y <= end.y
            && other_end.z <= end.z
    }

    /// Source: util/math/box3i.h:72-90
    pub fn intersects(&self, other: Box3i) -> bool {
        let end = self.end();
        let other_end = other.end();
        self.position.x < other_end.x
            && self.position.y < other_end.y
            && self.position.z < other_end.z
            && other.position.x < end.x
            && other.position.y < end.y
            && other.position.z < end.z
    }

    /// Grows the box by `m` on every side.
    /// Source: util/math/box3i.h:274-276
    pub fn padded(&self, m: i32) -> Box3i {
        Box3i::new(
            self.position - IVec3::splat(m),
            self.size + IVec3::splat(2 * m),
        )
    }

    /// Source: util/math/box3i.h:297-299
    pub fn scaled(&self, scale: i32) -> Box3i {
        Box3i::new(self.position * scale, self.size * scale)
    }

    /// Source: util/math/box3i.h:314-318
    pub fn clip(&mut self, lim: Box3i) {
        clip_range(
            &mut self.position.x,
            &mut self.size.x,
            lim.position.x,
            lim.size.x,
        );
        clip_range(
            &mut self.position.y,
            &mut self.size.y,
            lim.position.y,
            lim.size.y,
        );
        clip_range(
            &mut self.position.z,
            &mut self.size.z,
            lim.position.z,
            lim.size.z,
        );
    }

    /// Source: util/math/box3i.h:326-330
    pub fn clipped(&self, lim: Box3i) -> Box3i {
        let mut copy = *self;
        copy.clip(lim);
        copy
    }

    /// Source: util/math/box3i.h:338-345
    pub fn encloses(&self, other: Box3i) -> bool {
        let end = self.end();
        let other_end = other.end();
        self.position.x <= other.position.x
            && self.position.y <= other.position.y
            && self.position.z <= other.position.z
            && end.x >= other_end.x
            && end.y >= other_end.y
            && end.z >= other_end.z
    }

    /// Source: util/math/box3i.h:354-356
    pub fn is_empty(&self) -> bool {
        self.size.x <= 0 || self.size.y <= 0 || self.size.z <= 0
    }

    /// Source: util/math/box3i.h:358-370
    pub fn merge_with(&mut self, other: Box3i) {
        let min_pos = self.position.min(other.position);
        let max_pos = self.end().max(other.end());
        self.position = min_pos;
        self.size = max_pos - min_pos;
    }

    /// Iterates every contained position in ZXY order, matching the buffer layout.
    pub fn for_each_cell_zxy(&self, mut f: impl FnMut(IVec3)) {
        if !self.is_empty() {
            let end = self.end();
            for z in self.position.z..end.z {
                for x in self.position.x..end.x {
                    for y in self.position.y..end.y {
                        f(IVec3::new(x, y, z));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_point_excludes_the_upper_bound() {
        let b = Box3i::new(IVec3::ZERO, IVec3::splat(4));
        assert!(b.contains_point(IVec3::new(0, 0, 0)));
        assert!(b.contains_point(IVec3::new(3, 3, 3)));
        assert!(!b.contains_point(IVec3::new(4, 0, 0)));
        assert!(!b.contains_point(IVec3::new(-1, 0, 0)));
    }

    #[test]
    fn clipping_disjoint_boxes_yields_empty() {
        let a = Box3i::new(IVec3::ZERO, IVec3::splat(4));
        let b = Box3i::new(IVec3::splat(10), IVec3::splat(4));
        assert!(a.clipped(b).is_empty());
        assert!(!a.intersects(b));
    }

    #[test]
    fn clipping_overlap_keeps_the_intersection() {
        let a = Box3i::new(IVec3::ZERO, IVec3::splat(4));
        let b = Box3i::new(IVec3::splat(2), IVec3::splat(4));
        let c = a.clipped(b);
        assert_eq!(c.position, IVec3::splat(2));
        assert_eq!(c.size, IVec3::splat(2));
        assert!(a.intersects(b));
    }

    #[test]
    fn padded_grows_symmetrically() {
        let b = Box3i::new(IVec3::ZERO, IVec3::splat(4)).padded(2);
        assert_eq!(b.position, IVec3::splat(-2));
        assert_eq!(b.size, IVec3::splat(8));
    }

    #[test]
    fn from_center_extents_matches_min_max() {
        let b = Box3i::from_center_extents(IVec3::splat(5), IVec3::splat(2));
        assert_eq!(b, Box3i::from_min_max(IVec3::splat(3), IVec3::splat(7)));
    }

    #[test]
    fn merge_with_covers_both_boxes() {
        let mut a = Box3i::new(IVec3::ZERO, IVec3::splat(2));
        let b = Box3i::new(IVec3::splat(5), IVec3::splat(2));
        a.merge_with(b);
        assert!(a.contains_box(Box3i::new(IVec3::ZERO, IVec3::splat(2))));
        assert!(a.contains_box(b));
    }

    #[test]
    fn for_each_cell_visits_every_position_once() {
        let b = Box3i::new(IVec3::new(-1, 0, 2), IVec3::new(2, 3, 4));
        let mut count = 0u64;
        b.for_each_cell_zxy(|p| {
            assert!(b.contains_point(p));
            count += 1;
        });
        assert_eq!(count, b.volume());
    }

    #[test]
    fn empty_box_has_zero_volume_and_visits_nothing() {
        let b = Box3i::new(IVec3::ZERO, IVec3::new(0, 4, 4));
        assert!(b.is_empty());
        assert_eq!(b.volume(), 0);
        let mut visited = false;
        b.for_each_cell_zxy(|_| visited = true);
        assert!(!visited);
    }
}
