/*  search.js — wires the search page to /api/spaces/<slug>/query.
 *  v0: scope-filter clicks update the active row visually; real query
 *      requires an embedding from the viewer.
 */
(function () {
  const scope = document.getElementById("scopeList");
  if (!scope) return;
  scope.addEventListener("click", function (e) {
    const row = e.target.closest("[data-space]");
    if (!row) return;
    scope.querySelectorAll(".filter-row").forEach(r => r.classList.remove("on"));
    row.classList.add("on");
  });

  // Populate Saved-names from every visible space (just the first one for now).
  // Replace with /api/spaces/<slug>/named-objects per scope selection.
})();
