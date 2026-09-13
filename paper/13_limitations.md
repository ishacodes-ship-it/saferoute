# 13. Limitations [DRAFTED — known upfront, not dependent on results]

These are known at design time, independent of what the experiments end
up showing (contrast with §12 Discussion, which depends on results):

1. **Whitelisted environment features are OSM-tag proxies, not
   measurements.** `footfall_proxy` is street-intersection density, not a
   pedestrian count. `open_businesses` is static commercial-POI density,
   not a real-time open/closed signal. `lighting` has sparse OSM `lit=*`
   tag coverage. `cctv` is crowdsourced OSM `man_made=surveillance`
   tagging, not a camera registry. Full detail:
   `src/saferoute/features/environment_features.py`.

2. **US-only validation.** The risk model, routing behavior, and every
   number in this paper are validated on Chicago's open crime data. No
   claim is made about performance on the target deployment city (India,
   per the original PRD framing) absent that city's own data.

3. **Synthetic OD pairs in the counterfactual benchmark.** No open
   pedestrian-trip dataset exists for Chicago; the benchmark's
   origin/destination pairs are constructed via a bounded random walk from
   real held-out incident locations, not replayed from real trip logs. The
   incidents themselves are real; the trips around them are not.

4. **Pareto frontier approximation.** Route alternatives come from a
   $\lambda$-sweep with non-domination filtering, not an exact
   multi-objective shortest-path algorithm — this can miss frontier points
   in non-convex regions of (time, risk) space.

5. **Routing engine choice trades production performance for research
   flexibility.** The OSMnx/NetworkX routing layer is slower per query
   than a contraction-hierarchy engine (OSRM/GraphHopper); acceptable for
   single-city research-scale evaluation, not validated at
   production/national scale.

6. **The Mesa feedback-loop simulation is a mechanism demonstration, not
   a calibrated forecast.** No real footfall dataset exists to calibrate
   its parameters against; it shows whether a named qualitative mechanism
   appears under this project's cost function, nothing more.

7. **No causal claim.** Nothing in this paper claims the system prevents
   crime, that a route is objectively "safe," or that a location is
   inherently dangerous — see README.md's ethics/terminology section,
   which the explanation-generation code enforces programmatically for
   its own output (not for hand-written paper prose, which remains a
   human review responsibility).
