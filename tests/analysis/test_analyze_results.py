"""The result analyses behind the paper's claims about the verifier."""

import numpy as np

from sphyr.analyze_results import failure_kind, solid_mask, topology


def test_a_ring_has_one_piece_and_one_hole():
    ring = np.ones((4, 4), dtype=bool)
    ring[1:3, 1:3] = False

    assert topology(ring) == (1, 1)


def test_corner_contact_joins_pieces_but_does_not_close_a_hole():
    # Elements meeting at a corner share a node, so they are one piece; the
    # void squeezed between them still reaches the border, so it is no hole.
    diagonal = np.eye(3, dtype=bool)

    assert topology(diagonal) == (1, 0)


def test_densities_count_as_solid_from_one_half():
    assert solid_mask([["0.4", "0.5"], ["L", "0"]]).tolist() == [
        [False, True],
        [True, False],
    ]


def test_unreadable_grids_are_told_apart():
    truth = "0 1\n1 0\n0 0"
    record = lambda completion: {"ground_truth": truth, "completion": completion}

    assert failure_kind(record("I cannot.")) == "no grid"
    assert failure_kind(record("0 1\n1 0")) == "stops short"
    assert failure_kind(record("0 1\n1 0\n0 0\n1 1")) == "extra rows"
    assert failure_kind(record("0 1\n1 0 1\n0 0")) == "wrong row width"
