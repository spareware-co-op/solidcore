//! A linear sweep against a circle's normal turns the circle's frame a half-turn about
//! its X axis before making the cylinder. Upstream turns by `180 * PI / 180` radians,
//! whose binary64 sine is about 1.2e-16, not 0, so the cylinder of a whole circle on
//! X (1, 0, 0), Y (0, -1, 0) swept along +z came out, with its two circles, with Y
//! (0, 1, -1.2246467991473532e-16) where the exact half-turn is (0, 1, 0), and an arc's
//! cylinder (`fixes.linear_sweep_arc_is_cylinder`) the same. The half-turn is exact: Y
//! and Z negated, X and the origin unchanged. Every number below is compared exactly.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)] // test helpers; exact comparison is the point of the test

use solidcore::curve::curve::{Curve, CurveKind};
use solidcore::curve::ellipse::Ellipse;
use solidcore::curve::line::Line;
use solidcore::geom::axis2_placement::Axis2Placement;
use solidcore::geom::extent1d::Extent1d;
use solidcore::surface::cone::Cone;
use solidcore::surface::surface::SurfaceKind;
use solidcore::sweep::api_brep::create_linear_sweep;
use solidcore::{Brep, Kernel, Vec3};

/// Asserts `got` is `want` bit for bit (the sign of a zero aside).
fn assert_exact(got: Vec3, want: Vec3, what: &str) {
    assert!(
        got.x == want.x && got.y == want.y && got.z == want.z,
        "{what}: {got:?}, want exactly {want:?}"
    );
}

/// Asserts `got` is `frame` turned a half-turn about its X axis, exactly: the same origin
/// and X axis, Y and Z negated.
fn assert_half_turned(got: &Axis2Placement, frame: &Axis2Placement, what: &str) {
    assert_exact(got.origin(), frame.origin(), &format!("{what} origin"));
    assert_exact(got.x_axis(), frame.x_axis(), &format!("{what} x axis"));
    assert_exact(got.y_axis(), -frame.y_axis(), &format!("{what} y axis"));
    assert_exact(got.z_axis(), -frame.z_axis(), &format!("{what} z axis"));
}

/// The B-rep's cones and cylinders, in face order.
fn cylinders(b: &Brep) -> Vec<Cone> {
    b.face_list()
        .iter()
        .filter_map(
            |&f| match b.face(f).unwrap().surface.as_ref().unwrap().kind() {
                SurfaceKind::Cone(c) => Some((**c).clone()),
                _ => None,
            },
        )
        .collect()
}

/// The frames of the B-rep's circular edges, in edge order.
fn circle_frames(b: &Brep) -> Vec<Axis2Placement> {
    b.edge_list()
        .iter()
        .filter_map(
            |&e| match b.edge(e).unwrap().curve.as_ref().unwrap().kind() {
                CurveKind::Ellipse(c) => Some(c.canonical().position),
                _ => None,
            },
        )
        .collect()
}

fn seg(a: Vec3, b: Vec3) -> Curve {
    Curve::from(Line::create_line_segment(3, a, b, None).unwrap())
}

/// The frame of a round hole as a profile hands it to the sweep: X along the square's
/// first edge, Y so that the circle turns negatively about the sweep (+z).
fn hole_frame(centre: Vec3) -> Axis2Placement {
    Axis2Placement::new(centre, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0))
}

/// The whole circle of radius 0.5 on `frame`. (Not a quadratic NURBS circle:
/// the sweep reads that one's frame through `IsArc`, which rounds its centre and X axis
/// before any turn.)
fn whole_circle(k: &Kernel, frame: &Axis2Placement) -> Curve {
    let interval = Extent1d::from_min_max(0.0, 360.0);
    Curve::from(Ellipse::create_canonical_circle(k, frame, 0.5, Some(&interval), None).unwrap())
}

#[test]
fn the_half_turn_negates_y_and_z_exactly() {
    let frame = Axis2Placement::new(
        Vec3::new(0.25, -1.5, 3.0),
        Vec3::new(0.6, 0.8, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    let mut turned = frame;
    turned.half_turn_about_x();
    assert_half_turned(&turned, &frame, "half-turn");
    turned.half_turn_about_x();
    assert_eq!(turned, frame, "two half-turns");
}

#[test]
fn a_whole_circle_swept_against_its_normal_is_turned_exactly() {
    // a round hole in a square, swept 3 along +z: its circle's normal is -z
    for centre in [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.3, -0.7, 0.0)] {
        let case = format!("circle at {centre:?}");
        let mut k = Kernel::default();
        let frame = hole_frame(centre);
        let p = |x: f64, y: f64| Vec3::new(x, y, 0.0);
        let curves = [
            seg(p(-2.0, -2.0), p(2.0, -2.0)),
            seg(p(2.0, -2.0), p(2.0, 2.0)),
            seg(p(2.0, 2.0), p(-2.0, 2.0)),
            seg(p(-2.0, 2.0), p(-2.0, -2.0)),
            whole_circle(&k, &frame),
        ];
        let refs: Vec<Option<&Curve>> = curves.iter().map(Some).collect();
        let b = create_linear_sweep(&mut k, &refs, Vec3::Z, 3.0, true).unwrap();
        let cylinders = cylinders(&b);
        assert_eq!(cylinders.len(), 1, "{case}");
        assert_half_turned(&cylinders[0].canonical().origin, &frame, &case);
        // the two circles: on the cylinder's frame, at the profile and 3 above it
        let circles = circle_frames(&b);
        assert_eq!(circles.len(), 2, "{case}");
        let mut heights = Vec::new();
        for c in &circles {
            let mut at = frame;
            at.set_origin(Vec3::new(centre.x, centre.y, c.origin().z));
            assert_half_turned(c, &at, &format!("{case} circle"));
            heights.push(c.origin().z);
        }
        heights.sort_by(f64::total_cmp);
        assert_eq!(heights, [0.0, 3.0], "{case}");
    }
}

#[test]
fn an_arc_swept_against_its_normal_is_turned_exactly() {
    // the quarter disc of radius 4 about (4, 4) on a frame whose normal is +z, swept 50
    // against it: the frame is turned to the arc's end (90 degrees about its Z axis, as
    // before) and then a half-turn about its X axis, exactly
    let centre = Vec3::new(4.0, 4.0, 0.0);
    let frame = Axis2Placement::new(centre, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
    let mut turned = frame;
    turned.rotate_about_axis_at_point(90.0 * std::f64::consts::PI / 180.0, centre, frame.z_axis());
    let turned = Axis2Placement::new(turned.origin(), turned.x_axis(), turned.y_axis());
    for (dir, dist) in [(-Vec3::Z, 50.0), (Vec3::Z, -50.0)] {
        let case = format!("{dir:?} {dist}");
        let mut k = Kernel::default();
        k.fixes.linear_sweep_arc_is_cylinder = true;
        let interval = Extent1d::from_min_max(0.0, 90.0);
        let arc = Curve::from(
            Ellipse::create_canonical_circle(&k, &frame, 4.0, Some(&interval), None).unwrap(),
        );
        let l1 = seg(Vec3::new(4.0, 0.0, 0.0), centre);
        let l2 = seg(centre, Vec3::new(0.0, 4.0, 0.0));
        let b = create_linear_sweep(&mut k, &[Some(&arc), Some(&l1), Some(&l2)], dir, dist, true)
            .unwrap();
        let cylinders = cylinders(&b);
        assert_eq!(cylinders.len(), 1, "{case}");
        assert_half_turned(&cylinders[0].canonical().origin, &turned, &case);
    }
}
