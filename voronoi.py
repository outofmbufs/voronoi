# MIT License
#
# Copyright (c) 2020 Neil Webber
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# Voronoi diagrams on x/y grids.
#
# These algorithms assume a "grid" with the following interfaces:
#     [ NOTE: All of these requirements are met by a simple dict storing
#             point objects indexed by an (x, y) tuple. Any other grid
#             representation meeting these requirements will also work ]
#
# TERMINOLOGY: XY TUPLES
#  A grid square has an x and y coordinate, or an "xy tuple":
#    xy = (x, y)  and  x, y = xy
#
# POINT LOOKUP:
#
#  Let g be a grid.
#
#      p = g[xy]   obtains a point, p, at grid coordinate xy
#
# The code does not care anything about p except:
#   If xy is not within limits, p MUST be None OR an exception must be raised.
#   If xy is within grid limits, p MUST NOT be None.
#
# Class attribute OFFGRIDEXCEPTIONS defines the acceptable exceptions.
#
# In effect, g[xy] is used only to test whether xy is in the grid or not.
#
# COORDINATE MATH:
#
# Given x, y = xy, the following are assumed:
#
#    Neighbors:
#       Can be found as g[(x-1, y-1)], g[(x-1, y)], ... etc
#
#       ** Override the _neighbors() method if this is not the case.
#
#    Distance Metric:
#       Given xy1 and xy2, the default distance metric defining Voronoi cells
#       is the euclidean distance calculated from delta-X and delta-Y values.
#
#       ** Override _distmetric for other metrics.
#


class Voronoi:
    """Voronoi diagrams. See https://en.wikipedia.org/wiki/Voronoi_diagram"""

    # Exceptions that will be taken as meaning "off grid" if raised by g[xy]
    # Override this as necessary.
    OFFGRIDEXCEPTIONS = (KeyError, IndexError)

    # Default distance metric is euclidean distance, squared.
    # Squared because the algorithm doesn't care about the values
    # themselves, only about greater/less relationships, so omitting
    # the square root step doesn't matter, and is a bit faster.
    #
    # Override this method if a different distance metric is required.
    #
    def _distmetric(self, xy1, xy2):
        """Return distance metric between xy1, xy2."""
        dx = xy1[0] - xy2[0]
        dy = xy1[1] - xy2[1]
        return (dx * dx) + (dy * dy)

    # x,y deltas to each of eight neighboring coordinates
    NEIGHBORS_8 = ((-1, -1), (-1, 0), (-1, 1),
                   (0, -1), (0, 1),
                   (1, -1), (1, 0), (1, 1))

    # x,y deltas to each of four neighboring coordinates (no diags)
    NEIGHBORS_4 = ((-1, 0), (0, -1), (0, 1), (1, 0))

    # override this in a subclass or just set it, to switch neighbor behavior
    NEIGHBOR_RULE = NEIGHBORS_8

    def _neighbors(self, g, xy):
        """Return xy tuples for the neighbors of xy."""
        x, y = xy
        for dx, dy in self.NEIGHBOR_RULE:
            try:
                nb = g[(x + dx, y + dy)]
            except self.OFFGRIDEXCEPTIONS:
                nb = None
            if nb is not None:
                yield (x + dx, y + dy)

    def __init__(self, g, sites):
        """Create voronoi cells from the given sites.

        v = Voronoi(g, sites)
               g: a "grid" that can be indexed by (x, y) tuple
           sites: iterable of (x, y) tuples definined Voronoi sites (seeds).
        """

        # algorithm assumes no duplicated sites; enforce this w/useful msg
        if len(set(sites)) != len(sites):
            raise ValueError("Voronoi sites cannot contain duplicates")

        # This is a (grid-quantized) "growing circles" Voronoi algorithm.
        #
        # Let R(N) be a "ring" at distance metric N from a given site.
        # NOTE: It rarely looks like a true ring, it is clumps of squares.
        # This example shows 7x7 around a site labeled with N values:
        #
        #        18  13  10   9  10  13  18
        #        13   8   5   4   5   8  13
        #        10   5   2   1   2   5  10
        #         9   4   1   0   1   4   9
        #        10   5   2   1   2   5  10
        #        13   8   5   4   5   8  13
        #        18  13  10   9  10  13  18
        #
        # Starting with each site (the "0" in the example shown), the site
        # itself is assigned (obviously) to the Voronoi seeded by that site.
        #
        # For each Voronoi cell an active perimeter is maintained as the list
        # of squares immediately adjacent to the already-assigned squares.
        #
        # For example, this shows two Voronoi sites ("0"), each surrounded
        # by 8 active perimeter squares ("A"). Note that perimeters may
        # overlap as they do in here at A*:
        #
        #         .   .   .   .   A   A   A
        #         .   .   .   .   A   0   A
        #         .   .   A   A   A*  A   A
        #         .   .   A   0   A   .   .
        #         .   .   A   A   A   .   .
        #         .   .   .   .   .   .   .
        #         .   .   .   .   .   .   .
        #
        # Both cells will have an active perimeter list containing 8 squares,
        # with "A*" appearing in both lists.
        #
        # The algorithm iterates over increasing N values, examining the
        # active perimeter squares and assigning them to their cell when
        # their metric matches the current N. By definition this assigns
        # squares to their closest Voronoi site without needing to compare
        # their metric to other sites (which, by definition, can't have
        # reached this square yet with a lower N or it already would have
        # been assigned in a prior iteration of the loop).
        #
        # The obvious exception is a square like A* which is in the active
        # perimeter list of two sites in this example. It is equidistant from
        # both and fundamentally ambiguous. Ambiguous squares are assigned
        # to Voronoi cells based on the order of their sites as given.
        #
        # This ambiguity is one manifestation of quantization error and should
        # be thought of as a fundamental aliasing artifact imposed by sampling
        # a continuous domain (Voronoi concept) onto a finite grid.
        #
        # There are other quantization effects that arise. Note in particular
        # that no grid Voronoi algorithm can satisfy both of these criteria
        # at all times:
        #   1: Every square is assigned to its closest Voronoi site
        #   2: Every Voronoi cell is a contiguous blob.
        #
        # For an in-depth discussion of this, see: https://tinyurl.com/sw4tga7
        #

        # xy2s is (becomes) a mapping (i.e. a dict) from any xy to its
        # corresponding Voronoi cell (denoted by the xy of its site).
        # Every site belongs to itself as a start:
        xy2s = {xy: xy for xy in sites}

        # Ambiguous squares are recorded here. If xy could have been
        # assigned to more than one equidistant site, then
        #         ambig[xy] = set of xy1, xy2, ...
        ambig = {}

        # The per-site active perimeters, stored in an inverted structure:
        #   perims_by_N:         dict indexed by distmetric, containing...
        #   perims_by_N[dm]      dict, indexed by xy, containing...
        #   perims_by_N[dm][xy]  list of sites that are at distance dm from xy

        perims_by_N = {}

        # helper to add neighbors of xy as being in perim of site s
        # NOTE: this takes liberal advantage of access to enclosing scope
        def _perims_add_nb(xy, s):
            for nb in self._neighbors(g, xy):
                if nb not in xy2s:
                    dm = self._distmetric(nb, s)
                    if dm not in perims_by_N:
                        perims_by_N[dm] = {}
                    xysN = perims_by_N[dm]
                    if nb not in xysN:
                        xysN[nb] = []
                    xysites = xysN[nb]
                    if s not in xysites:
                        xysites.append(s)

        # start the initial site perimeters
        for s in sites:
            _perims_add_nb(s, s)

        while perims_by_N:
            # take the smallest known distance metric
            # ?? there might be a more clever way, is it always
            #    the first in the (ordered-by-insertions) dict?
            N = min(perims_by_N.keys())
            for xy, xysites in list(perims_by_N[N].items()):
                if xy not in xy2s:     # i.e., only do if not already claimed

                    # always simply take the first (whether only, or "of N")
                    s = xysites[0]
                    xy2s[xy] = s

                    # but keep track of the ambiguities, just because
                    if len(xysites) > 1:
                        ambig[xy] = xysites

                    # NOTE: by definition the neighbors cannot be at this
                    # same N; therefore, although this modifies perims_by_N
                    # it does not modify perims_by_N[N] used in this loop
                    _perims_add_nb(xy, s)

            del perims_by_N[N]

        # xy2s now maps every xy to a site; construct reverse map too
        # (from a site to a set of squares that are the cell for that site)
        site2xys = {s: set() for s in sites}
        for xy in xy2s:
            site2xys[xy2s[xy]].add(xy)

        self.__grid = g
        self.__xy2s = xy2s
        self.__site2xys = site2xys
        self.ambiguous = ambig

    @property
    def sites(self):
        return self.__site2xys.keys()

    # allow "for site in v"
    def __iter__(self):
        return self.__site2xys.__iter__()

    # given any xy, return the site of its Voronoi cell
    def xy_to_site(self, xy):
        return self.__xy2s[xy]

    # given ANY xy, return all the xys in its Voronoi cell
    def cellxys(self, xy):
        return list(self.__site2xys[self.__xy2s[xy]])

    # Lloyd's "relaxation" algorithm to smooth Voronoi polygons.
    # NOTE: THIS CREATES A NEW Voronoi
    #
    # Relocates each site to the centroid of its cell and recomputes the
    # Voronoi cells, which become more "regular" as a result. Useful
    # to get "smoother" diagrams when appropriate.
    # See https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
    #
    def lloyd(self):
        """Returns a NEW Voronoi computed by Lloyd's algorithm from self."""
        newsites = []
        for site in self.sites:
            xtotal = 0
            ytotal = 0
            n = 0
            for cellxy in self.cellxys(site):
                xtotal += cellxy[0]
                ytotal += cellxy[1]
                n += 1
            if n == 0:
                newsites.append(site)
            else:
                x = int((xtotal/n) + 0.5)
                y = int((ytotal/n) + 0.5)
                newsites.append((x, y))
        return Voronoi(self.__grid, newsites)

    # simple-minded string representation of a Voronoi
    # No attempt is made here to handle the "map coloring" problem
    def vcellsstr(v):
        str = ""
        sites_to_char = {}
        reps = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        nth = 0
        xy_to_char = {}
        xmin = xmax = ymin = ymax = None
        for s in v:
            sites_to_char[s] = reps[nth % len(reps)]
            nth += 1
            # this is hokey, but avoids assuming anything about xy ranges...
            # discover from them examining all the xys in all the cells
            for xy in v.cellxys(s):
                if xmin is None or xy[0] < xmin:
                    xmin = xy[0]
                if xmax is None or xy[0] > xmax:
                    xmax = xy[0]
                if ymin is None or xy[1] < ymin:
                    ymin = xy[1]
                if ymax is None or xy[1] > ymax:
                    ymax = xy[1]

        for y in range(ymax, ymin-1, -1):
            line = ""
            for x in range(xmin, xmax+1):
                site = v.xy_to_site((x, y))
                ch = sites_to_char[site]
                if (x, y) == site:
                    ch = ch.lower()
                line += ch
            str += line + '\n'
        return str


if __name__ == "__main__":
    # Unit Tests
    import unittest
    import random

    class TestMethods(unittest.TestCase):

        def makemap(self, xsize=100, ysize=100, defval='.'):
            return {(x, y): defval for x in range(xsize) for y in range(ysize)}

        def fillfromstrings(self, d, *strings):
            y = 0
            for s in strings:
                for x in range(len(s)):
                    d[(x, y)] = s[x]
                y += 1

        def test_vor1(self):
            size = 8             # MUST BE EVEN
            g = self.makemap(size, size)
            sitexys = ((0, 0),
                       (0, size-1),
                       (size-1, 0),
                       (size-1, size-1)
                       )
            v = Voronoi(g, sitexys)

            # given the sites in the four corners, and size even, each
            # cell should be a square half the size in each direction

            for s in v:
                xys = v.cellxys(s)
                self.assertEqual(len(xys), (size//2)*(size//2))
                for xy in xys:
                    self.assertTrue(abs(xy[0] - s[0]) < size//2)
                    self.assertTrue(abs(xy[1] - s[1]) < size//2)

        def test_vor2(self):
            size = 8
            g = self.makemap(size, size)
            sitexys = ((0, 0), (0, 1), (1, 0), (1, 1))
            v = Voronoi(g, sitexys)

            # the size of the (0, 0) cell should be one -- just the site itself
            xys = v.cellxys((0, 0))
            self.assertEqual(len(xys), 1)
            self.assertEqual(xys[0], (0, 0))

            # the size of the (0, 1) and (1, 0) should be size-1 and
            # they should run "along their respective sides"
            xys = v.cellxys((1, 0))
            self.assertEqual(len(xys), size-1)
            coords = set(xys)
            self.assertEqual(coords, set([(x, 0) for x in range(1, size)]))

            xys = v.cellxys((0, 1))
            self.assertEqual(len(xys), size-1)
            coords = set(xys)
            self.assertEqual(coords, set([(0, y) for y in range(1, size)]))

            # the size of the (1, 1) cell should be all the rest.
            xys = v.cellxys((1, 1))
            self.assertEqual(len(xys), (size*size) - ((2*(size-1)) + 1))

            # finally, though this is somewhat redundant at this point,
            # the union of all the Voronoi cells should be... all
            vxys = set()
            for s in v:
                for xy in v.cellxys(s):
                    self.assertTrue(xy not in vxys)
                    vxys.add(xy)
            self.assertEqual(len(vxys), size*size)

        def test_vor3(self):
            size = 25
            g = self.makemap(size, size)
            v = Voronoi(g, [(x, y) for x in range(size) for y in range(size)])
            self.assertEqual(len(v.sites), size*size)
            # every cell should have just one square
            for s in v:
                xys = v.cellxys(s)
                self.assertEqual(len(xys), 1)

        def test_vor4(self):
            g = self.makemap(8, 5)
            v = Voronoi(g, ((2, 2), (6, 1)))

            # this has just been hand-constructed as the correct answer
            correct = \
                "AAAAABBB\n" + \
                "AAAAABBB\n" + \
                "AAaAABBB\n" + \
                "AAAABBbB\n" + \
                "AAAABBBB\n"

            self.assertEqual(v.vcellsstr(), correct)

        def test_vor5(self):
            size = 50

            # these test vectors are collected from some randomly generated
            # cases but also some hand-culled "problematic" test cases.
            test_vectors = (
                ((0, 0), (size-1, size-1)),
                ((4, 3), (12, 0), (0, 6)),
                ((25, 11), (33, 8), (21, 14)),
                ((19, 0), (13, 15), (18, 7)),
                ((18, 7), (19, 0), (13, 15)),
                ((11, 37), (6, 19), (14, 38), (25, 11), (11, 21),
                 (17, 18), (11, 3), (33, 8), (21, 14)),
                ((8, 47), (6, 8), (21, 3), (45, 46), (5, 24),
                 (39, 4), (35, 48), (10, 19), (29, 2), (9, 43)),
                ((38, 47), (12, 4), (23, 36), (30, 27), (5, 16),
                 (29, 46), (28, 7), (34, 18), (28, 20), (16, 28)),
                ((5, 40), (30, 1), (26, 42), (29, 8), (36, 3),
                 (49, 48), (32, 29), (16, 13), (34, 10), (14, 6)),
                ((7, 47), (24, 29), (45, 41), (38, 4), (2, 10),
                 (48, 34), (17, 15), (42, 39), (41, 23), (6, 1)),
                ((19, 13), (43, 19), (41, 35), (7, 18), (26, 35),
                 (8, 43), (37, 35), (38, 29), (39, 7), (31, 47)),
                ((20, 35), (7, 17), (26, 31), (40, 25), (21, 48),
                 (11, 26), (21, 33), (25, 43), (5, 5), (4, 12)),
                ((4, 7), (38, 11), (18, 39), (7, 10), (25, 4),
                 (9, 35), (35, 17), (21, 38), (32, 9), (1, 30)),
                ((2, 38), (23, 23), (27, 11), (11, 0), (7, 28),
                 (29, 3), (46, 12), (45, 9), (49, 5), (20, 13)),
                ((33, 17), (2, 12), (33, 32), (14, 15), (12, 5),
                 (18, 47), (16, 25), (45, 3), (36, 42), (16, 20)),
                ((23, 4), (12, 32), (15, 17), (3, 15), (12, 30),
                 (21, 7), (25, 27), (2, 28), (44, 32), (7, 16)))
            for onetest in test_vectors:
                g = self.makemap(size, size)
                self.brutevv(g, Voronoi(g, onetest))

        def brutevv(self, g, v, slack=1.01):
            # brute-force verify every point is indeed closest to the
            # voronoi seed for the one it was put into.
            # HOWEVER, allow a small amount of slack (1%, arbitrary).
            # See Voronoi code for extensive discussion this issue.

            for xy in g.keys():
                xysite = v.xy_to_site(xy)
                d0 = v._distmetric(xysite, xy)

                for othersite in v.sites:
                    if othersite != xysite:
                        d1 = v._distmetric(othersite, xy)
                        d1 *= slack
                        s = ""
                        s = f"xy={xy}"
                        s += f" d0={d0} d1={d1}"
                        s += f" xysite={xysite}"
                        s += f"othersite={othersite}"
                        s += f" {v.sites}"
                        self.assertTrue(d0 <= d1, s)

        def test_vor6(self):
            # same idea as test_vor5 but with randomness
            for nth in range(5):
                size = random.randrange(30, 300)
                nsites = max(size // random.randrange(3, 50), 1)
                g = self.makemap(size, size)
                sites = random.sample(set(g.keys()), nsites)
                self.brutevv(g, Voronoi(g, sites))

        def test_lloyd1(self):
            # very simple-minded test, start with a 5x5 with a site in
            # the corner; after lloyd site should be in center.
            # TODO: Obviously need more/better lloyd tests.
            size = 5
            g = self.makemap(size, size)
            v = Voronoi(g, ((0, 0),))
            v2 = v.lloyd()
            xysite = v2.xy_to_site((0, 0))
            self.assertEqual(xysite, (size//2, size//2))

if __name__ == "__main__":
    unittest.main()
