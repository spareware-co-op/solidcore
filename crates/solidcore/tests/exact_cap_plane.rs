//! A flat face made on the loops of its curves lies on their plane exactly when every
//! point of the curves does. The plane's normal is Newell's sum over points of the loops
//! and its base point their average. Upstream sums Newell's terms in absolute coordinates,
//! `(y1 - y2) (z1 + z2)` and so on, and the rounding of the differences is multiplied by
//! the loops' height: the top cap of the cylinder of radius 0.5 and height 1 on the origin
//! came out on a plane with normal (2.1e-16, 0, 1) through z = 0.9999999999999999, while
//! its bottom cap, at height 0, was exact. Summed about a point of the loops, every term
//! of a loop at one height is exactly zero, and the plane is that height's exactly.
//! Every number below is compared exactly. (A circle's points are its rational B-spline's,
//! whose evaluation rounds a coordinate the circle holds constant, so a cap on a circle
//! at a height other than 0 can still be off its plane by a rounding: not this fix.)

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)] // test helpers; exact comparison is the point of the test

use solidcore::curve::curve::Curve;
use solidcore::curve::line::Line;
use solidcore::geom::axis2_placement::Axis2Placement;
use solidcore::primitive::api_primitives::create_cylinder;
use solidcore::surface::surface::SurfaceKind;
use solidcore::sweep::api_brep::create_linear_sweep;
use solidcore::{Brep, Kernel, Point3, Vec3};

/// The frames of the B-rep's planes, in face order.
fn planes(b: &Brep) -> Vec<Axis2Placement> {
    b.face_list()
        .iter()
        .filter_map(
            |&f| match b.face(f).unwrap().surface.as_ref().unwrap().kind() {
                SurfaceKind::Plane(p) => Some(*p.position()),
                _ => None,
            },
        )
        .collect()
}

/// Asserts that `plane` is square to z exactly, through height `z`.
fn assert_level(plane: &Axis2Placement, z: f64, what: &str) {
    assert!(
        plane.origin().z == z,
        "{what}: origin {:?}, want z exactly {z}",
        plane.origin()
    );
    assert!(
        plane.x_axis().z == 0.0 && plane.y_axis().z == 0.0,
        "{what}: axes {:?} {:?} leave the plane z = {z}",
        plane.x_axis(),
        plane.y_axis()
    );
    let n = plane.z_axis();
    assert!(
        n.x == 0.0 && n.y == 0.0 && n.z.abs() == 1.0,
        "{what}: normal {n:?}, want exactly (0, 0, 1) or its opposite"
    );
}

/// The heights of the B-rep's flat faces, which must be level, low to high.
fn level_heights(b: &Brep, what: &str) -> Vec<f64> {
    let mut heights = Vec::new();
    for plane in planes(b) {
        let z = plane.origin().z;
        assert_level(&plane, z, what);
        heights.push(z);
    }
    heights.sort_by(f64::total_cmp);
    heights
}

#[test]
fn a_cylinder_has_its_caps_square_to_its_axis() {
    // the cylinder of radius 0.5 and height 1 on the origin: its circles' points are all
    // at heights 0 and 1 exactly, and so must its caps be.
    let mut k = Kernel::default();
    let b = create_cylinder(&mut k, Point3::new(0.0, 0.0, 0.0), 0.5, 1.0).unwrap();
    assert_eq!(level_heights(&b, "cylinder"), [0.0, 1.0]);
}

#[test]
fn a_polygon_swept_square_to_its_plane_has_level_caps() {
    // a triangle and a square hole in it, at heights whose sums round, swept 3 along +z
    for z in [0.0, 1.0, 0.1, -2.7] {
        let what = format!("profile at z = {z}");
        let mut k = Kernel::default();
        let p = |x: f64, y: f64| Vec3::new(x, y, z);
        let seg = |a: Vec3, b: Vec3| Curve::from(Line::create_line_segment(3, a, b, None).unwrap());
        let corners = [p(-1.3, -0.7), p(2.1, -0.45), p(0.35, 1.9)];
        let hole = [p(0.1, 0.1), p(0.1, 0.3), p(0.3, 0.3), p(0.3, 0.1)];
        let mut curves = Vec::new();
        for ring in [&corners[..], &hole[..]] {
            for (i, &a) in ring.iter().enumerate() {
                curves.push(seg(a, ring[(i + 1) % ring.len()]));
            }
        }
        let refs: Vec<Option<&Curve>> = curves.iter().map(Some).collect();
        let b = create_linear_sweep(&mut k, &refs, Vec3::Z, 3.0, true).unwrap();
        // the seven sides are upright; the two caps are level, at z and z + 3
        let caps: Vec<Axis2Placement> = planes(&b)
            .into_iter()
            .filter(|plane| plane.z_axis().z.abs() > 0.5)
            .collect();
        assert_eq!(caps.len(), 2, "{what}");
        for cap in &caps {
            assert_level(cap, cap.origin().z, &what);
        }
        let mut heights: Vec<f64> = caps.iter().map(|c| c.origin().z).collect();
        heights.sort_by(f64::total_cmp);
        assert_eq!(heights, [z, z + 3.0], "{what}");
    }
}
