(() => {
  const $ = (selector) => document.querySelector(selector);
  const state = {
    sources: [],
    page: 0,
    pageSize: 40,
    totalPosts: 0,
    charts: {},
    activeJobId: null,
    currentJobStatus: null,
    selectedMapPost: null,
    reportLoaded: false,
  };
  const palette = ["#003DA5", "#55759f", "#8097b8", "#FFB81C", "#aebbd0", "#365b91", "#d4dce8", "#6681a9", "#e4c46f", "#8b9cb6", "#476b9f", "#c6ceda"];

  const STATIC_MODE = Boolean(window.HWM_STATIC_MODE);
  const STATIC_BASE = window.HWM_STATIC_BASE || "./";
  let staticSnapshot = null;

  function staticUrl(path) {
    return STATIC_MODE ? new URL(path.replace(/^\/+/, ""), new URL(STATIC_BASE, window.location.href)).href : path;
  }

  async function staticApi(path) {
    if (!staticSnapshot) {
      const response = await fetch(staticUrl("snapshot.json"), { cache: "no-store" });
      if (!response.ok) throw new Error("Could not load this dashboard snapshot.");
      staticSnapshot = await response.json();
    }
    const url = new URL(path, window.location.href);
    if (url.pathname.endsWith("/api/overview")) return staticSnapshot.overview;
    if (url.pathname.endsWith("/api/sources")) return { sources: staticSnapshot.sources };
    if (url.pathname.endsWith("/api/classification-status")) return staticSnapshot.classification;
    if (url.pathname.endsWith("/api/report/status")) return staticSnapshot.report;
    if (url.pathname.endsWith("/api/job")) return { job: null };
    if (url.pathname.endsWith("/api/posts")) {
      const params = url.searchParams;
      let posts = staticSnapshot.posts;
      const q = (params.get("q") || "").toLocaleLowerCase();
      const classification = params.get("classification") || "";
      const division = params.get("division") || "";
      const theme = params.get("theme") || "";
      const sentiment = params.get("sentiment") || "";
      const period = params.get("period") || "";
      const mapped = params.get("mapped") === "1";
      if (q) posts = posts.filter(post => `${post.title || ""} ${post.latest_selftext || ""}`.toLocaleLowerCase().includes(q));
      if (classification) posts = posts.filter(post => classification === "classified"
        ? post.classification_status === "classified" : post.classification_status !== "classified");
      if (division === "any") posts = posts.filter(post => (post.classifications || []).length > 0);
      else if (division) posts = posts.filter(post => (post.classifications || []).some(hit => hit.key === division));
      if (theme) posts = posts.filter(post => post.theme_name === theme);
      if (sentiment) posts = posts.filter(post => post.sentiment <= -0.05 ? sentiment === "Negative"
        : post.sentiment >= 0.05 ? sentiment === "Positive" : sentiment === "Neutral");
      if (period) posts = posts.filter(post => (post.creation_time || "").slice(0, period.length) === period);
      if (mapped) posts = posts.filter(post => (post.locations || []).some(location => location.has_point));
      const count = posts.length;
      const page = Math.max(0, Number(params.get("page") || 0));
      const limit = Math.max(1, Number(params.get("limit") || 40));
      posts = posts.slice(page * limit, (page + 1) * limit);
      return { posts, count, page, limit };
    }
    throw new Error("This is a read-only static snapshot. Update data from the local dashboard.");
  }

  async function api(path, options = {}) {
    if (STATIC_MODE) return staticApi(path);
    const response = await fetch(path, { cache: "no-store", ...options,
      headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) } });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
    return result;
  }

  function queryString(extra = {}) {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(extra)) {
      if (value !== "" && value !== null && value !== undefined) params.set(key, String(value));
    }
    return params.toString();
  }

  function showError(message) {
    const banner = $("#errorBanner");
    banner.textContent = message;
    banner.classList.remove("hidden");
  }

  function clearError() {
    $("#errorBanner").classList.add("hidden");
  }

  async function loadSources() {
    const data = await api("/api/sources");
    state.sources = data.sources.filter((source) => source.enabled);
    const list = $("#sourceList");
    list.replaceChildren();
    if (!state.sources.length) {
      list.appendChild(safeText("p", "source-empty", "No subreddits are listed for future scraping."));
      $("#scrapeButton").disabled = true;
      return;
    }
    $("#scrapeButton").disabled = STATIC_MODE || state.currentJobStatus === "running";
    state.sources.forEach((source) => {
      const chip = document.createElement("div");
      chip.className = "source-chip";
      chip.appendChild(safeText("span", "", `r/${source.subreddit}`));
      const remove = safeText("button", "source-remove", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove r/${source.subreddit} from future scrapes`);
      remove.title = "Remove from future scrapes";
      if (!STATIC_MODE) {
        remove.addEventListener("click", () => removeSource(source.subreddit));
        chip.appendChild(remove);
      }
      list.appendChild(chip);
    });
  }

  async function loadOverview() {
    const data = await api(`/api/overview?${queryString()}`);
    $("#metricPosts").textContent = Number(data.total_posts).toLocaleString();
    const communities = (data.subreddit_labels || []).map((name) => `r/${String(name).replace(/^r\//i, "")}`);
    const visibleCommunities = communities.slice(0, 3);
    const communityText = communities.length > 3
      ? `${visibleCommunities.join(", ")}, …`
      : visibleCommunities.join(", ");
    $("#metricPostsFoot").textContent = communities.length
      ? `Scraped from ${communityText}`
      : "No subreddit sources in this dataset";
    $("#metricPostsFoot").title = communities.join(", ");
    $("#metricSentiment").textContent = Number(data.average_sentiment).toFixed(2);
    $("#metricMapped").textContent = Number(data.location_posts).toLocaleString();
    $("#metricClassified").textContent = Number(data.classified_posts).toLocaleString();
    $("#metricClassifiedFoot").textContent = `${Number(data.unclassified_posts).toLocaleString()} not classified yet`;
    $("#overviewScope").textContent = STATIC_MODE
      ? "Across all posts in this snapshot" : "Across all posts in the local database";
    $("#lastUpdated").textContent = STATIC_MODE && staticSnapshot?.generated_at
      ? `Snapshot updated ${new Date(staticSnapshot.generated_at).toLocaleString()}`
      : `Updated ${new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
    populateThemeFilter(data.theme_options || data.theme_labels || []);
    populatePeriodFilter(data.trend_labels || []);
    drawTrend(data);
    drawDonut("sentimentChart", data.sentiment_labels, data.sentiment_counts,
      ["#a94646", "#cbd2dc", "#55759f"], "Sentiment");
    drawDonut("themeChart", data.theme_labels, data.theme_counts, palette, "Posts");
    drawDivisionBars(data.division_labels, data.division_counts, data.division_keys);
    const topicStatus = data.topic_status || {};
    $("#topicStatus").textContent = topicStatus.refreshed_at
      ? (topicStatus.clustering_scope === "full_initial_baseline" ? "Initial UMAP/HDBSCAN baseline" : "Incremental themes") +
        " · " + Number(topicStatus.clusters || 0) + " clusters · " +
        Number(topicStatus.assignments_kept || 0).toLocaleString() + " existing assignments kept · " +
        Number(topicStatus.posts_checked || 0).toLocaleString() + " new/changed posts checked · refreshed " +
        new Date(topicStatus.refreshed_at).toLocaleString()
      : "Using saved notebook labels until the first local topic refresh.";
    $("#topicRefreshButton").disabled = STATIC_MODE || state.currentJobStatus === "running";
    const divisionPosts = data.division_counts.reduce((sum, count) => sum + count, 0);
    $("#divisionNote").textContent = divisionPosts
      ? `Counts total ${divisionPosts.toLocaleString()} label assignments; posts may have more than one.`
      : "No classified division labels in this filter yet.";
  }

  function replaceChart(id, configuration) {
    if (state.charts[id]) state.charts[id].destroy();
    Chart.defaults.font.family = "Fira Sans, sans-serif";
    Chart.defaults.color = "#667085";
    const canvas = document.getElementById(id);
    if (!canvas || !window.Chart) return;
    state.charts[id] = new Chart(canvas, configuration);
  }

  function drawTrend(data) {
    replaceChart("trendChart", {
      type: "bar",
      data: { labels: data.trend_labels, datasets: [
        { type: "bar", label: "Posts", data: data.trend_posts, backgroundColor: "rgba(0,61,165,.62)", borderRadius: 4, yAxisID: "y" },
        { type: "line", label: "Average sentiment", data: data.trend_sentiment, borderColor: "#d29a16", backgroundColor: "#d29a16", pointRadius: 2.5, pointHoverRadius: 4, tension: .28, yAxisID: "y1" },
      ] },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        onClick: (event, elements, chart) => {
          const hit = elements[0];
          const index = hit ? hit.index : chart.getElementsAtEventForMode(event, "index", { intersect: false }, true)[0]?.index;
          if (index == null) return;
          setPostFilters({ period: chart.data.labels[index] });
        },
        plugins: { legend: { labels: { usePointStyle: true, boxWidth: 7, font: { size: 11 } } } },
        scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 12, maxRotation: 0, color: "#819194", font: { size: 10 } } },
          y: { beginAtZero: true, grid: { color: "#edf1ef" }, ticks: { precision: 0, color: "#819194", font: { size: 10 } } },
          y1: { position: "right", min: -1, max: 1, grid: { drawOnChartArea: false }, ticks: { color: "#819194", font: { size: 10 } } } } },
    });
  }

  function drawDonut(id, labels, values, colors, name) {
    replaceChart(id, { type: "doughnut", data: { labels, datasets: [{ data: values,
      backgroundColor: labels.map((_, i) => colors[i % colors.length]), borderColor: "#fff", borderWidth: 3, hoverOffset: 5 }] },
      options: { responsive: true, maintainAspectRatio: false, cutout: "61%",
        onClick: (_event, elements) => {
          const hit = elements[0];
          if (!hit) return;
          setPostFilters(id === "themeChart" ? { theme: labels[hit.index] } : { sentiment: labels[hit.index] });
        },
        plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 7, padding: 12, font: { size: 10 } } },
          tooltip: { callbacks: { label(context) { const total = context.dataset.data.reduce((a, b) => a + b, 0) || 1;
            return `${context.label}: ${context.parsed.toLocaleString()} (${(context.parsed / total * 100).toFixed(1)}%)`; } } } } } });
  }

  function drawDivisionBars(labels, values, keys = []) {
    replaceChart("divisionChart", { type: "bar", data: { labels, datasets: [{ label: "Post labels", data: values,
      backgroundColor: "rgba(0,61,165,.75)", borderRadius: 4, maxBarThickness: 22 }] },
      options: { indexAxis: "y", responsive: true, maintainAspectRatio: false,
        onClick: (_event, elements) => {
          const hit = elements[0];
          if (hit && keys[hit.index]) setPostFilters({ division: keys[hit.index] });
        },
        plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, grid: { color: "#edf1ef" }, ticks: { precision: 0 } },
          y: { grid: { display: false }, ticks: { font: { size: 10 } } } } } });
  }

  function selectedFilters() {
    return {
      q: $("#postSearch").value.trim(),
      classification: $("#classificationFilter").value,
      division: $("#divisionFilter").value,
      theme: $("#themeFilter").value,
      sentiment: $("#sentimentFilter").value,
      period: $("#periodFilter").value,
      mapped: $("#mappedOnly").checked ? 1 : 0,
      limit: state.pageSize,
      page: state.page,
    };
  }

  function setPostFilters(filters) {
    $("#postSearch").value = "";
    $("#classificationFilter").value = "";
    $("#divisionFilter").value = "any";
    $("#themeFilter").value = "";
    $("#sentimentFilter").value = "";
    $("#periodFilter").value = "";
    $("#mappedOnly").checked = false;
    if (Object.hasOwn(filters, "division")) $("#divisionFilter").value = filters.division;
    if (Object.hasOwn(filters, "theme")) $("#themeFilter").value = filters.theme;
    if (Object.hasOwn(filters, "sentiment")) $("#sentimentFilter").value = filters.sentiment;
    if (Object.hasOwn(filters, "period")) $("#periodFilter").value = filters.period;
    updateActiveFilterCount();
    state.page = 0;
    activateTab("posts");
  }

  function updateActiveFilterCount() {
    const count = [
      $("#classificationFilter").value,
      $("#divisionFilter").value,
      $("#themeFilter").value,
      $("#sentimentFilter").value,
      $("#periodFilter").value,
      $("#mappedOnly").checked,
      $("#postSearch").value.trim(),
    ].filter(Boolean).length;
    const badge = $("#activeFilterCount");
    badge.textContent = count ? String(count) : "";
    badge.hidden = count === 0;
  }

  function fillSelect(select, labels, defaultLabel, selectedValue = "") {
    select.replaceChildren();
    const all = document.createElement("option");
    all.value = "";
    all.textContent = defaultLabel;
    select.appendChild(all);
    labels.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.value;
      option.textContent = item.label;
      select.appendChild(option);
    });
    if ([...select.options].some((option) => option.value === selectedValue)) select.value = selectedValue;
  }

  function populateThemeFilter(themes) {
    const select = $("#themeFilter");
    const selected = select.value;
    fillSelect(select, [...new Set(themes)].filter((name) => name && name !== "Unclustered")
      .map((name) => ({ value: name, label: name })), "All themes", selected);
  }

  function populatePeriodFilter(periods) {
    const select = $("#periodFilter");
    const selected = select.value;
    fillSelect(select, periods.map((period) => ({ value: period, label: period })), "All periods", selected);
  }

  async function loadPosts(append = false) {
    const extra = selectedFilters();
    const data = await api(`/api/posts?${queryString(extra)}`);
    state.totalPosts = data.count;
    $("#postCount").textContent = `${Number(data.count).toLocaleString()} post${data.count === 1 ? "" : "s"} match`;
    if (!append) $("#postList").replaceChildren();
    if (!data.posts.length && !append) {
      $("#postList").innerHTML = '<div class="empty-state">No posts match these filters.</div>';
    }
    data.posts.forEach(renderPost);
    const hasMore = (data.page + 1) * data.limit < data.count;
    $("#loadMoreButton").classList.toggle("hidden", !hasMore);
  }

  function safeText(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text;
    return element;
  }

  function renderPost(post) {
    const card = document.createElement("article");
    card.className = "post-card";
    const top = document.createElement("div");
    top.className = "post-topline";
    const description = document.createElement("div");
    description.className = "post-description";
    description.appendChild(safeText("h3", "post-title", post.title || "(No title)"));
    const date = post.creation_time ? new Date(post.creation_time).toLocaleString() : "Date unavailable";
    description.appendChild(safeText("div", "post-meta", `r/${post.subreddit} · ${date} · score ${post.score ?? 0}`));
    const badges = document.createElement("div");
    badges.className = "post-badges";
    if (post.classification_status === "classified") {
      if (post.classifications.length) {
        post.classifications.forEach((hit) => {
          const badge = safeText("span", "badge", hit.name);
          if (hit.reason) badge.title = `${hit.confidence == null ? "" : `Confidence ${Number(hit.confidence).toFixed(2)} · `}${hit.reason}`;
          badges.appendChild(badge);
        });
      } else badges.appendChild(safeText("span", "badge", "Classified · no HWS division"));
    } else badges.appendChild(safeText("span", "badge unclassified", "Not classified"));
    if (post.theme_name && post.theme_name !== "Unclustered") {
      badges.appendChild(safeText("span", "badge theme", "Theme · " + post.theme_name));
    }
    (post.locations || []).forEach((location) => badges.appendChild(safeText("span",
      `badge ${location.has_point ? "location" : ""}`, `${location.name}${location.has_point ? "" : " · no pin yet"}`)));
    badges.appendChild(safeText("span", "badge", `Sentiment ${Number(post.sentiment).toFixed(2)}`));
    description.appendChild(badges);
    top.appendChild(description);

    const actions = document.createElement("div");
    actions.className = "post-actions";
    const expand = document.createElement("button");
    expand.className = "mini-button expand-button";
    expand.type = "button";
    expand.textContent = "View post";
    actions.appendChild(expand);
    const pointLocations = (post.locations || []).filter((location) => location.has_point);
    if (pointLocations.length) {
      const mapButton = document.createElement("button");
      mapButton.className = "mini-button";
      mapButton.type = "button";
      mapButton.textContent = "⌖ Map pin";
      mapButton.addEventListener("click", () => focusMap(post.id, pointLocations.map((item) => item.name)));
      actions.appendChild(mapButton);
    }
    top.appendChild(actions);
    card.appendChild(top);

    const details = document.createElement("div");
    details.className = "post-details hidden";
    details.appendChild(safeText("p", "post-body", post.latest_selftext || "(No selftext in the latest saved record.)"));
    const imageUrls = [...new Set((post.images || []).map((item) => typeof item === "string" ? item : item && item.url)
      .filter((url) => typeof url === "string" && isRedditImageUrl(url)))];
    if (imageUrls.length) {
      const gallery = document.createElement("div");
      gallery.className = "post-image-gallery";
      imageUrls.forEach((url, index) => {
        const link = document.createElement("a");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.setAttribute("aria-label", `Open post image ${index + 1}`);
        const image = document.createElement("img");
        image.dataset.src = url;
        image.alt = `Image ${index + 1} attached to this Reddit post`;
        image.decoding = "async";
        image.referrerPolicy = "no-referrer";
        image.addEventListener("error", () => {
          image.remove();
          link.textContent = "Open image in new tab ↗";
          link.classList.add("image-fallback");
        }, { once: true });
        link.appendChild(image);
        gallery.appendChild(link);
      });
      details.appendChild(gallery);
    }
    post.classifications.forEach((hit) => {
      if (hit.reason) details.appendChild(safeText("p", "class-hit", `${hit.name} · ${hit.reason}`));
    });
    const footer = document.createElement("div");
    footer.className = "post-footer";
    if (/^https:\/\/(www\.)?reddit\.com\//i.test(post.permalink || "")) {
      const source = document.createElement("a");
      source.href = post.permalink;
      source.target = "_blank";
      source.rel = "noopener noreferrer";
      source.textContent = "Open Reddit post ↗";
      footer.appendChild(source);
    }
    footer.appendChild(safeText("span", "subtle", `ID ${post.id}`));
    details.appendChild(footer);
    card.appendChild(details);
    expand.addEventListener("click", () => {
      const opening = details.classList.contains("hidden");
      details.classList.toggle("hidden", !opening);
      expand.textContent = opening ? "Hide post" : "View post";
      if (opening) {
        requestAnimationFrame(() => details.querySelectorAll("img[data-src]").forEach((image) => {
          image.src = image.dataset.src;
          image.removeAttribute("data-src");
        }));
      }
    });
    $("#postList").appendChild(card);
  }

  function isRedditImageUrl(value) {
    try {
      const url = new URL(value);
      return url.protocol === "https:" && url.hostname.toLowerCase().endsWith(".redd.it");
    } catch {
      return false;
    }
  }

  async function loadCoverage() {
    const data = await api(`/api/classification-status?${queryString()}`);
    $("#coverageCount").textContent = `${Number(data.classified).toLocaleString()} / ${Number(data.total).toLocaleString()}`;
    $("#coverageText").textContent = "posts have a saved classifier result";
    const percent = data.total ? Math.round(data.classified / data.total * 100) : 0;
    $("#coveragePercent").textContent = `${percent}% complete · ${Number(data.unclassified).toLocaleString()} remaining`;
    $("#coverageProgress").style.width = `${percent}%`;
    const running = state.currentJobStatus === "running";
    $("#classifyButton").disabled = STATIC_MODE || running;
    $("#classifyButton").textContent = data.classified && data.unclassified
      ? `Resume classification · ${Number(data.unclassified).toLocaleString()} left`
      : data.unclassified === 0 ? "All database posts classified" : "Classify unclassified posts";
    $("#classifyButton").disabled = STATIC_MODE || running || data.unclassified === 0;
  }

  function focusMap(postId, locations) {
    state.selectedMapPost = postId;
    $("#mapSelection").textContent = `Highlighting ${locations.join(" + ")} for this post.`;
    activateTab("map");
    reloadMap(postId);
  }

  function reloadMap(postId = state.selectedMapPost) {
    const query = new URLSearchParams();
    if (postId) query.set("post_id", postId);
    query.set("v", Date.now().toString());
    $("#mapFrame").src = `${staticUrl(STATIC_MODE ? "map/portal-map.html" : "/gis/portal-map.html")}?${query.toString()}`;
  }

  function activateTab(name) {
    document.querySelectorAll(".tab").forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("active", active);
      if (active) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `${name}Panel`));
    if (window.matchMedia("(max-width: 650px)").matches) {
      document.querySelector(`#${name}Panel`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (name === "map" && !state.selectedMapPost) reloadMap(null);
    if (name === "report") loadReportStatus();
    if (name === "posts") {
      loadPosts(false).catch((error) => showError(error.message));
      loadCoverage().catch((error) => showError(error.message));
    }
  }

  async function startJob(kind) {
    clearError();
    const endpoint = kind === "scrape" ? "/api/jobs/scrape"
      : kind === "topics" ? "/api/jobs/topics"
        : kind === "report" ? "/api/jobs/report" : "/api/jobs/classify";
    const payload = kind === "scrape" ? { sources: state.sources.map((source) => source.subreddit).join(",") } : {};
    const response = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
    state.activeJobId = response.job.id;
    state.currentJobStatus = response.job.status;
    paintJob(response.job);
    pollJob();
  }

  function paintJob(job) {
    if (!job) return;
    const banner = $("#jobBanner");
    banner.classList.remove("hidden");
    $("#jobTitle").textContent = job.kind === "scrape" ? "Scrape, map, and topic refresh"
      : job.kind === "topics" ? "Local topic refresh"
        : job.kind === "report" ? "Gemini research report" : "HWS classification";
    $("#jobMessage").textContent = job.message || "Working in the background…";
    $("#scrapeButton").disabled = job.status === "running" || state.sources.length === 0;
    ["#classifyButton", "#topicRefreshButton", "#reportGenerateButton"].forEach((selector) => {
      $(selector).disabled = job.status === "running";
    });
    const total = Number(job.progress_total || 0);
    const done = Number(job.progress_done || 0);
    $("#jobProgress").style.width = total ? `${Math.min(100, done / total * 100)}%` : (job.status === "running" ? "30%" : "100%");
    $("#jobProgressText").textContent = total ? `${done.toLocaleString()} / ${total.toLocaleString()}` : job.status;
  }

  async function pollJob() {
    if (!state.activeJobId) return;
    try {
      const result = await api(`/api/job?id=${encodeURIComponent(state.activeJobId)}`);
      const job = result.job;
      if (!job) return;
      state.currentJobStatus = job.status;
      paintJob(job);
      if (job.status === "running") {
        setTimeout(pollJob, 1700);
        return;
      }
      const previousKind = job.kind;
      state.activeJobId = null;
      if (previousKind === "scrape" && job.status === "complete") reloadMap(state.selectedMapPost);
      await Promise.all([loadOverview(), loadPosts(false), loadCoverage()]);
      if (previousKind === "report" && job.status === "complete") showGeneratedReport(true);
      if (job.status === "failed") showError(job.message);
    } catch (error) {
      showError(error.message);
      setTimeout(pollJob, 3000);
    }
  }

  function showGeneratedReport(forceReload = false, generatedAt = null) {
    const frame = $("#reportFrame");
    const download = $("#reportDownloadButton");
    const pdf = $("#reportPdfButton");
    const wrap = $("#reportFrameWrap");
    if (forceReload || !state.reportLoaded) {
      const url = `${staticUrl(STATIC_MODE ? "report/latest.html" : "/reports/latest.html")}?v=${Date.now()}`;
      frame.src = url;
      download.href = url;
      state.reportLoaded = true;
    }
    wrap.classList.remove("hidden");
    download.classList.remove("hidden");
    pdf.classList.remove("hidden");
    $("#reportStatus").textContent = generatedAt
      ? `Latest report generated ${new Date(generatedAt).toLocaleString()}. Download PDF opens the browser print dialog; choose Save as PDF.`
      : "Report generated. Download PDF opens the browser print dialog; choose Save as PDF.";
  }

  async function loadReportStatus() {
    try {
      const result = await api("/api/report/status");
      if (result.available) showGeneratedReport(false, result.generated_at);
    } catch (_error) {
      // A not-yet-started dashboard report is an optional view; leave its prompt visible.
    }
  }

  async function refreshEverything() {
    clearError();
    try {
      await Promise.all([loadOverview(), loadPosts(false), loadCoverage()]);
      if ($("#mapPanel").classList.contains("active")) reloadMap(state.selectedMapPost);
    } catch (error) {
      showError(error.message);
    }
  }

  function debounce(fn, wait = 350) {
    let timer;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
  }

  function initDivisionFilter() {
    const divisionList = [
      ["basic_needs", "Basic Needs"], ["care", "CARE"], ["caps", "CAPS"],
      ["swift", "SWIFT Mobile Crisis Response"], ["case_management", "Case Management"],
      ["sdrc", "Student Disability Resource Center"], ["shs", "Student Health Services"],
      ["ucpd", "Police & Campus Safety"], ["the_well", "The Well"],
    ];
    const select = $("#divisionFilter");
    const current = select.value;
    if (!select.querySelector('option[value="any"]')) {
      const any = document.createElement("option");
      any.value = "any";
      any.textContent = "Any division";
      select.insertBefore(any, select.firstChild);
    }
    divisionList.forEach(([key, name]) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = name;
      select.appendChild(option);
    });
    select.value = current || "any";
  }

  async function initialize() {
    if (STATIC_MODE) {
      document.body.classList.add("static-snapshot");
      $(".live-indicator").textContent = "Read-only snapshot";
      $(".intro-copy").textContent = "A static snapshot of public Reddit discussions, themes, and campus locations.";
      $(".footer").firstElementChild.textContent = "Static snapshot · Updated from the local database";
      ["#refreshButton", "#topicRefreshButton", "#scrapeButton", "#classifyButton", "#reportGenerateButton", "#reloadMapButton"]
        .forEach(selector => { const button = $(selector); if (button) button.disabled = true; });
      $("#newSource").disabled = true;
      $("#addSourceForm button[type=submit]").disabled = true;
      $("#reportStatus").textContent = "This static copy cannot generate a new report. Use the local dashboard, then export a fresh snapshot.";
      $("#reportGenerateButton").title = "Generate the report locally, then export a new static snapshot.";
      $("#mapSelection").textContent = "Choose a mapped post to center and highlight its pin. Map pins are from this export.";
    }
    initDivisionFilter();
    try {
      if (STATIC_MODE) await staticApi("/api/overview");
      await loadSources();
      await refreshEverything();
    } catch (error) {
      showError(error.message);
    }
    document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
    $("#filterToggle").addEventListener("click", () => {
      const button = $("#filterToggle");
      const expanded = button.getAttribute("aria-expanded") !== "true";
      button.setAttribute("aria-expanded", String(expanded));
      $("#postFilterControls").classList.toggle("is-open", expanded);
    });
    updateActiveFilterCount();
    $("#postSearch").addEventListener("input", debounce(() => {
      updateActiveFilterCount();
      state.page = 0;
      loadPosts(false).catch((error) => showError(error.message));
    }));
    ["#classificationFilter", "#divisionFilter", "#themeFilter", "#sentimentFilter", "#periodFilter", "#mappedOnly"].forEach((selector) =>
      $(selector).addEventListener("change", () => {
        updateActiveFilterCount();
        state.page = 0;
        loadPosts(false).catch((error) => showError(error.message));
      }));
    $("#loadMoreButton").addEventListener("click", () => {
      state.page += 1;
      loadPosts(true).catch((error) => showError(error.message));
    });
    $("#addSourceForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const input = $("#newSource");
      try {
        await api("/api/sources", { method: "POST", body: JSON.stringify({ subreddit: input.value }) });
        input.value = "";
        await loadSources();
      } catch (error) { showError(error.message); }
    });
    $("#scrapeButton").addEventListener("click", () => startJob("scrape").catch((error) => showError(error.message)));
    $("#classifyButton").addEventListener("click", () => startJob("classify").catch((error) => showError(error.message)));
    $("#topicRefreshButton").addEventListener("click", () => startJob("topics").catch((error) => showError(error.message)));
    $("#reportGenerateButton").addEventListener("click", () => startJob("report").catch((error) => showError(error.message)));
    $("#reportPdfButton").addEventListener("click", () => {
      const frame = $("#reportFrame");
      if (frame.contentWindow) frame.contentWindow.print();
    });
    $("#refreshButton").addEventListener("click", refreshEverything);
    $("#reloadMapButton").addEventListener("click", () => {
      state.selectedMapPost = null;
      $("#mapSelection").textContent = "Choose a mapped post to center and highlight its pin.";
      reloadMap(null);
    });
  }

  async function removeSource(subreddit) {
    clearError();
    try {
      await api(`/api/sources/${encodeURIComponent(subreddit)}`, { method: "DELETE" });
      await loadSources();
    } catch (error) { showError(error.message); }
  }

  document.addEventListener("DOMContentLoaded", initialize);
})();
