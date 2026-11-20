//! A straight edge swept along a straight direction sweeps a parallelogram: a plane. The
//! linear sweep made a profile line's side a plane only when the line was square to the
//! sweep; swept obliquely, upstream made the side a surface of extrusion, which the record
//! writes as a bilinear spline net with spline edges (the rounding of its far row can twist
//! the net off the plane). The side is the planar face on the parallelogram's four lines,
//! as each cap is the planar face on its loop, so a prism swept obliquely has planes and
//! lines only, and its corners are the profile lines' ends and those ends moved by the
//! sweep, exactly. (A line segment keeps its start, unit direction and length, so its end
//! is evaluated: the segment from (1, 0, 0) to (0, 1, 0) ends at (-2.2e-16, 1.0000000000000002,
//! 0), on a sweep square to it too. The corners are compared with the ends as the lines give
//! them.) A sweep by zero sweeps no area: the oblique line's parallelogram then has two sides
//! of no length and is dropped, as the square line's plane of no height is.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)] // test helpers; exact comparison is the point of the test

use solidcore::curve::curve::{Curve, CurveKind};
use solidcore::curve::line::Line;
use solidcore::geom::attribute::{AttributeStore, ObjectKey};
use solidcore::mass::integrator::properties::{brep_volume, PropertyAttributes};
use solidcore::surface::surface::SurfaceKind;
use solidcore::sweep::api_brep::create_linear_sweep;
use solidcore::topology::brep::validation::{check_brep, CheckLevel};
use solidcore::{Brep, Kernel, Vec3};

fn seg(a: Vec3, b: Vec3) -> Curve {
    Curve::from(Line::create_line_segment(3, a, b, None).unwrap())
}

fn volume(k: &Kernel, b: &mut Brep) -> f64 {
    let store = AttributeStore::default();
    let attrs = PropertyAttributes {
        store: &store,
        key_of: &|_| ObjectKey(0),
    };
    brep_volume(k, b, 1e-7, &attrs).unwrap()
}

fn same(p: &Vec3, q: &Vec3) -> bool {
    p.x == q.x && p.y == q.y && p.z == q.z
}

/// The triangle `t` swept `distance` along `direction`, capped, with its lines' ends as
/// the lines give them.
fn prism(k: &mut Kernel, t: [Vec3; 3], direction: Vec3, distance: f64) -> (Brep, Vec<Vec3>) {
    let curves = [seg(t[0], t[1]), seg(t[1], t[2]), seg(t[2], t[0])];
    let mut ends = Vec::new();
    for c in &curves {
        let e = c.ends(k).unwrap();
        ends.extend([e.start, e.end]);
    }
    let refs: Vec<Option<&Curve>> = curves.iter().map(Some).collect();
    let b = create_linear_sweep(k, &refs, direction, distance, true).unwrap();
    (b, ends)
}

/// Asserts the prism swept by `v` (the sweep's own unit direction times its length) has
/// five plane faces, nine line edges and six vertices, each exactly a profile line's end
/// or that end moved by `v`, every one of the profile's corners with its corner exactly `v`
/// along; that it is valid; and that it holds `want` of volume. Returns the vertices.
fn assert_plane_prism(
    b: &mut Brep,
    k: &mut Kernel,
    ends: &[Vec3],
    v: Vec3,
    want: f64,
    case: &str,
) -> Vec<Vec3> {
    let kinds: Vec<&str> = b
        .face_list()
        .iter()
        .map(
            |&f| match b.face(f).unwrap().surface.as_ref().unwrap().kind() {
                SurfaceKind::Plane(_) => "plane",
                SurfaceKind::SurfOfExtrusion(_) => "extrusion",
                _ => "other",
            },
        )
        .collect();
    assert_eq!(kinds, ["plane"; 5], "{case}: faces");
    let edges: Vec<bool> = b
        .edge_list()
        .iter()
        .map(|&e| {
            matches!(
                b.edge(e).unwrap().curve.as_ref().unwrap().kind(),
                CurveKind::Line(_)
            )
        })
        .collect();
    assert_eq!(edges, [true; 9], "{case}: every edge a line");
    let seen: Vec<Vec3> = b
        .vertex_list()
        .iter()
        .map(|&id| b.vertex(id).unwrap().point)
        .collect();
    assert_eq!(seen.len(), 6, "{case}: vertices {seen:?}");
    let mut bottom = 0;
    for p in &seen {
        if ends.iter().any(|e| same(p, e)) {
            bottom += 1;
            let q = *p + v;
            assert!(
                seen.iter().any(|s| same(s, &q)),
                "{case}: no vertex exactly at {q:?}, {p:?} moved by the sweep, in {seen:?}"
            );
        } else {
            assert!(
                ends.iter().any(|e| same(p, &(*e + v))),
                "{case}: the vertex {p:?} is no line's end, nor one moved by the sweep"
            );
        }
    }
    assert_eq!(bottom, 3, "{case}: the profile's corners in {seen:?}");
    let lines = check_brep(k, b, CheckLevel::Level2).unwrap().lines();
    assert!(lines.is_empty(), "{case}: check_brep: {lines:?}");
    let got = volume(k, b);
    assert!(
        (got - want).abs() <= 1e-9,
        "{case}: volume {got}, want {want}"
    );
    seen
}

#[test]
fn a_prism_swept_obliquely_has_plane_sides_on_its_exact_corners() {
    // the triangle on z = 0 swept 3 along (1, 2, 1): every side oblique; the sweep moves
    // a point by the unit direction times 3, and the volume is the triangle's area times
    // the height that moves, 1/2 * 3 / sqrt(6)
    let t = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let direction = Vec3::new(1.0, 2.0, 1.0);
    let mut k = Kernel::default();
    let (mut b, ends) = prism(&mut k, t, direction, 3.0);
    let v = direction.unitize().unwrap() * 3.0;
    let want = 0.5 * 3.0 / 6.0_f64.sqrt();
    assert_plane_prism(&mut b, &mut k, &ends, v, want, "z = 0 by (1, 2, 1)");
}

#[test]
fn a_tilted_prism_swept_along_an_axis_has_plane_sides_on_its_exact_corners() {
    // the triangle (0, 0, 0), (1, 0, 1), (0, 1, 0) swept 2 along +z: two sides oblique,
    // one (along y) square; every corner is a whole number, the top ones 2 above the
    // bottom ones, and the volume is the triangle's area seen from above times 2
    let t = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let mut k = Kernel::default();
    let (mut b, ends) = prism(&mut k, t, Vec3::Z, 2.0);
    let v = Vec3::new(0.0, 0.0, 2.0);
    let seen = assert_plane_prism(&mut b, &mut k, &ends, v, 1.0, "tilted by +z");
    for c in t.iter().copied().chain(t.iter().map(|&p| p + v)) {
        assert!(
            seen.iter().any(|p| same(p, &c)),
            "tilted by +z: no vertex exactly at {c:?} in {seen:?}"
        );
    }
}

/// The line from the origin to `end` swept `distance` along +z, capped or not, with the
/// sweep statuses passed on or not.
fn line_swept(end: Vec3, distance: f64, cap: bool, pass_on: bool) -> solidcore::Result<Brep> {
    let mut k = Kernel::default();
    k.fixes.sweep_statuses_pass_on = pass_on;
    let line = seg(Vec3::ZERO, end);
    create_linear_sweep(&mut k, &[Some(&line)], Vec3::Z, distance, cap)
}

/// The square line (0, 0, 0) to (1, 0, 0) and the oblique line (0, 0, 0) to (1, 0, 1).
fn lines() -> [(&'static str, Vec3); 2] {
    [
        ("square", Vec3::new(1.0, 0.0, 0.0)),
        ("oblique", Vec3::new(1.0, 0.0, 1.0)),
    ]
}

#[test]
fn a_line_swept_obliquely_by_zero_has_no_side_as_a_square_line_has_none() {
    // a zero distance sweeps no area: the square line's plane of no height fails and is
    // dropped as upstream drops it, so the sweep succeeds with no face; the oblique line's
    // parallelogram, whose sides from its end and back to its start have no length, is
    // dropped the same; a single open line takes no caps, so caps change nothing
    for cap in [false, true] {
        for (name, end) in lines() {
            let b = line_swept(end, 0.0, cap, false)
                .unwrap_or_else(|e| panic!("{name} line by 0, caps {cap}: {e:?}"));
            let counts = (
                b.face_list().len(),
                b.edge_list().len(),
                b.vertex_list().len(),
            );
            assert_eq!(counts, (0, 0, 0), "{name} line by 0, caps {cap}");
        }
    }
}

#[test]
fn a_line_swept_obliquely_by_zero_passes_its_status_on_as_a_square_line_does() {
    // with the sweep statuses passed on, the dropped side's failure is the sweep's, for
    // the oblique line as for the square one
    for (name, end) in lines() {
        let got = line_swept(end, 0.0, false, true);
        assert!(
            got.is_err(),
            "{name} line by 0, statuses passed on: no error"
        );
    }
}
