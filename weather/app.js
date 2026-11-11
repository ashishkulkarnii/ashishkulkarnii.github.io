/* Data source adapters: Open-Meteo is active; a future nowcast source plugs in here. */
const IST = "Asia/Kolkata";
const now = new Date();
now.setMinutes(0, 0, 0);
const DEFAULT_HOURS = 48;
const NOW_INDEX = 0;
const sources = { openMeteo: { id:"open-meteo", label:"Open-Meteo forecast model", cadence:"hourly" } };
const defaultPoint = { lat: 12.9716, lng: 77.5946, label: "Bengaluru" };
const defaultBounds = [[12.834, 77.460], [13.142, 77.784]];

const state = { point: defaultPoint, bounds: defaultBounds, geometry: null, approximateArea:false, regionKind:"Area", radiusKm: 10, coverageThreshold: .1, highContrast:false, index: NOW_INDEX, playing: true, mode: "area", source:sources.openMeteo, window:{ start:new Date(now), end:new Date(now.getTime()+(DEFAULT_HOURS-1)*3600000) }, weather:{ cells:[], times:[], cache:new Set(), loading:false, error:null }, weatherRequest:0 };
const el = Object.fromEntries(["placeSearch","searchButton","searchResults","placeLabel","areaTab","coordinateTab","areaPanel","coordinatePanel","coordinateFields","latitude","longitude","radius","radiusField","coverageThreshold","highContrast","advancedLink","advancedClose","advancedPanel","metricsTitle","modeBadge","locationLabel","timeline","playButton","simulationDate","timelineTitle","timelineMode","dataMode","liveStatus","chance","coverage","intensity","peak","accumulation","startLabel","endLabel","clickPrompt","searchHint","metricNote"].map(id => [id, document.getElementById(id)]));

const map = L.map("map", { zoomControl: false, doubleClickZoom: false }).setView([20.5937, 78.9629], 5);
L.control.zoom({ position: "bottomright" }).addTo(map);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "© OpenStreetMap contributors",
  // OSM requires an identifiable referrer. Serve this app on HTTP(S), not file://.
  referrerPolicy: "strict-origin-when-cross-origin"
}).addTo(map);

const rainLayer = L.layerGroup().addTo(map);
let selectionCircle;
let selectionMarker;

function formatTime(date, includeDate = true) {
  return new Intl.DateTimeFormat("en-IN", { timeZone: IST, weekday: includeDate ? "short" : undefined, day: includeDate ? "numeric" : undefined, month: includeDate ? "short" : undefined, hour: "numeric", minute: "2-digit", hour12: true }).format(date).replace(/am|pm/i, value => value.toUpperCase());
}
function istDayStamp(date) { const parts=new Intl.DateTimeFormat("en-CA",{timeZone:IST,year:"numeric",month:"2-digit",day:"2-digit"}).formatToParts(date).reduce((out,part)=>({ ...out,[part.type]:part.value}),{}); return Date.UTC(Number(parts.year),Number(parts.month)-1,Number(parts.day)); }
function relativeDayLabel(date) { const days=Math.round((istDayStamp(date)-istDayStamp(state.window.start))/86400000); return days ? ` + ${days} day${Math.abs(days)===1 ? "" : "s"}` : ""; }
function hourKey(date) { return date.toISOString().slice(0,13); }
function windowTimes() { const times=[]; for(let time=new Date(state.window.start);time<=state.window.end;time=new Date(time.getTime()+3600000)) times.push(hourKey(time)); return times; }
function timeForIndex(index) { const stamp=state.weather.times[index]; return stamp ? new Date(`${stamp}:00:00Z`) : new Date(state.window.start.getTime() + index * 60 * 60_000); }
function timeMode() { return state.source.label; }
function updateTimelineLabels() { const count=state.weather.times.length || windowTimes().length; el.timeline.max=Math.max(0,count-1); el.startLabel.textContent=formatTime(timeForIndex(0),false); el.endLabel.textContent=`${formatTime(timeForIndex(count-1),false)}${relativeDayLabel(timeForIndex(count-1))}`; }
function updateInputs() {
  el.latitude.value = state.point.lat.toFixed(5); el.longitude.value = state.point.lng.toFixed(5);
  el.radius.value = state.radiusKm;
  el.locationLabel.textContent = state.mode === "coordinate" ? `${state.point.label} · ${state.radiusKm} km radius` : `${state.point.label} · ${state.approximateArea ? "approximate PIN extent" : "region boundary"}`;
  el.metricsTitle.textContent=state.mode === "coordinate" ? "Coordinate conditions" : state.regionKind === "PIN code" ? `${state.point.label} PIN code conditions` : `${state.point.label} conditions`;
}
function updateSelection(fit = false) {
  const latlng = [state.point.lat, state.point.lng];
  if (selectionCircle) selectionCircle.remove(); if (selectionMarker) selectionMarker.remove();
  selectionCircle = state.mode === "coordinate"
    ? L.circle(latlng, { radius: state.radiusKm * 1000, color: "#254fc9", weight: 2, fillColor: "#4c75e8", fillOpacity: .08, dashArray: "5 5" }).addTo(map)
    : state.geometry ? L.geoJSON(state.geometry, { style: { color: "#254fc9", weight: 2, fillColor: "#4c75e8", fillOpacity: .08, dashArray:state.approximateArea ? "5 5" : undefined } }).addTo(map) : null;
  selectionMarker = L.circleMarker(latlng, { radius: 6, color: "#fff", weight: 2, fillColor: "#1e54d6", fillOpacity: 1 }).bindTooltip(state.point.label, { direction: "top", offset: [0,-8] }).addTo(map);
  if (fit && selectionCircle) map.fitBounds(selectionCircle.getBounds(), { padding: [40, 40], maxZoom: 10 });
}
function colorFor(value) { if (value < .25) return null; if (value < 2) return "#45c6ff"; if (value < 5) return "#2876ed"; if (value < 8) return "#6d45d8"; if (value < 12) return "#d63bb8"; return "#ff553e"; }
function dimensionsKm(bounds) { const [[south,west],[north,east]]=bounds; return { width:distanceKm((south+north)/2,west,(south+north)/2,east), height:distanceKm(south,(west+east)/2,north,(west+east)/2) }; }
function adaptiveGridDimensions(bounds,targetKm,minCells,maxCells) { const {width,height}=dimensionsKm(bounds); return { rows:Math.max(minCells,Math.min(maxCells,Math.ceil(height/targetKm))), columns:Math.max(minCells,Math.min(maxCells,Math.ceil(width/targetKm))) }; }
function buildRainGrid() {
  rainLayer.clearLayers();
  const mapBounds = state.mode === "coordinate"
    ? [[state.point.lat - state.radiusKm / 110 * 1.25, state.point.lng - state.radiusKm / (110 * Math.cos(state.point.lat * Math.PI/180)) * 1.25], [state.point.lat + state.radiusKm / 110 * 1.25, state.point.lng + state.radiusKm / (110 * Math.cos(state.point.lat * Math.PI/180)) * 1.25]]
    : state.bounds;
  const south=mapBounds[0][0], west=mapBounds[0][1], north=mapBounds[1][0], east=mapBounds[1][1];
  // A dense, adaptive display mesh is interpolated only for legibility; it does not affect coverage.
  const {rows,columns}=adaptiveGridDimensions(mapBounds,1.25,36,48); const dLat=(north-south)/rows, dLng=(east-west)/columns;
  for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
    const lat = south + (row + .5) * dLat, lng = west + (col + .5) * dLng;
    const inside=state.mode === "coordinate" ? distanceKm(state.point.lat,state.point.lng,lat,lng)<=state.radiusKm : pointInGeometry(lat,lng,state.geometry);
    if (!inside) continue;
    const rect = L.rectangle([[lat-dLat/2,lng-dLng/2],[lat+dLat/2,lng+dLng/2]], { stroke:false, fill:true, fillOpacity:0, interactive:false, className:"rain-cell" }).addTo(rainLayer);
    rect._rainPoint = { lat, lng };
  }
}
function analysisBounds() {
  return state.mode === "coordinate"
    ? [[state.point.lat - state.radiusKm / 110, state.point.lng - state.radiusKm / (110 * Math.cos(state.point.lat * Math.PI/180))], [state.point.lat + state.radiusKm / 110, state.point.lng + state.radiusKm / (110 * Math.cos(state.point.lat * Math.PI/180))]]
    : state.bounds;
}
/*
  This is the single adapter that production data must replace. It returns the source's
  native precipitation cells and values; coverage never reads from the visual overlay.
*/
function coverageGridDefinition() {
  const bounds=analysisBounds(); const [[south,west],[north,east]]=bounds;
  // Fixed 99-point analytical grid inside the selected region's tight enclosing bounds.
  // For a wide region, this is 11 columns × 9 rows; for a tall region, it is 9 columns × 11 rows.
  const {width,height}=dimensionsKm(bounds); const rows=width >= height ? 9 : 11, columns=width >= height ? 11 : 9;
  const dLat=(north-south)/rows, dLng=(east-west)/columns;
  const cells=[];
  for(let row=0;row<rows;row++) for(let column=0;column<columns;column++) {
    const lat=south+(row+.5)*dLat, lng=west+(column+.5)*dLng;
    cells.push({ bounds:[[lat-dLat/2,lng-dLng/2],[lat+dLat/2,lng+dLng/2]], lat, lng });
  }
  return { cells, rows, columns };
}
function nearestIncomingCell(lat,lng) { return state.weather.cells.reduce((nearest,cell)=>!nearest || distanceKm(lat,lng,cell.lat,cell.lng)<distanceKm(lat,lng,nearest.lat,nearest.lng)?cell:nearest,null); }
function weatherValue(cell,field) { return cell?.[field]?.get(state.weather.times[state.index]) ?? 0; }
function interpolatedPrecipitationAt(lat,lng) {
  if (!state.weather.cells.length) return null;
  const closest=[];
  state.weather.cells.forEach(cell=>{ const candidate={cell,distance:distanceKm(lat,lng,cell.lat,cell.lng)}; const index=closest.findIndex(item=>candidate.distance<item.distance); if(index===-1) closest.push(candidate); else closest.splice(index,0,candidate); if(closest.length>4) closest.pop(); });
  if (closest[0].distance < .001) return weatherValue(closest[0].cell,"precipitation");
  let weighted=0, weights=0;
  closest.forEach(({cell,distance})=>{ const weight=1/(distance*distance); weighted+=weatherValue(cell,"precipitation")*weight; weights+=weight; });
  return weights ? weighted/weights : null;
}
async function loadOpenMeteoForecast() {
  const request=++state.weatherRequest; const gridDefinition=coverageGridDefinition(); const grid=gridDefinition.cells;
  const signature=grid.map(cell=>`${cell.lat.toFixed(5)},${cell.lng.toFixed(5)}`).join("|");
  const previous=state.weather.signature === signature ? state.weather : null;
  const requested=windowTimes(); const missing=requested.filter(time=>!previous?.cache?.has(time));
  const cells=previous ? previous.cells : grid.map(cell=>({ ...cell, precipitation:new Map(), probability:new Map() }));
  state.weather={ cells, times:requested, grid:gridDefinition, signature, cache:previous?.cache || new Set(), loading:missing.length>0, error:null }; updateTimelineLabels(); renderMetrics();
  if (!missing.length) { render(); return; }
  try {
    const ranges=[]; missing.forEach(time=>{ const last=ranges.at(-1); if(last && new Date(`${time}:00:00Z`).getTime()===last.end.getTime()+3600000) last.end=new Date(`${time}:00:00Z`); else ranges.push({start:new Date(`${time}:00:00Z`),end:new Date(`${time}:00:00Z`)}); });
    for (const range of ranges) {
      const chunks=Array.from({length:Math.ceil(grid.length/100)},(_,index)=>grid.slice(index*100,index*100+100));
      const responses=await Promise.all(chunks.map(async chunk=>{ const latitudes=chunk.map(cell=>cell.lat.toFixed(5)).join(","), longitudes=chunk.map(cell=>cell.lng.toFixed(5)).join(","); const url=`https://api.open-meteo.com/v1/forecast?latitude=${latitudes}&longitude=${longitudes}&hourly=precipitation,precipitation_probability&start_hour=${hourKey(range.start)}:00&end_hour=${hourKey(range.end)}:00&timezone=UTC`; const response=await fetch(url); if(!response.ok) throw new Error(`Open-Meteo request failed (${response.status})`); const payload=await response.json(); return Array.isArray(payload)?payload:[payload]; }));
      if(request!==state.weatherRequest) return;
      const locations=responses.flat();
      locations.forEach((location,index)=>location.hourly?.time?.forEach((time,timeIndex)=>{ const key=time.slice(0,13); cells[index].precipitation.set(key,location.hourly.precipitation?.[timeIndex] ?? 0); cells[index].probability.set(key,location.hourly.precipitation_probability?.[timeIndex] ?? 0); state.weather.cache.add(key); }));
    }
    if(request!==state.weatherRequest) return;
    state.weather={ ...state.weather, loading:false, error:null }; state.index=Math.min(state.index,Math.max(0,requested.length-1)); updateTimelineLabels(); render();
  } catch(error) { if(request!==state.weatherRequest) return; state.weather={ ...state.weather, loading:false, error:error.message }; renderMetrics(); }
}
function updateRainLayer() {
  rainLayer.eachLayer(rect => {
    const precipitation=interpolatedPrecipitationAt(rect._rainPoint.lat,rect._rainPoint.lng);
    const displayValue=state.highContrast && precipitation !== null ? precipitation * 2.7 : precipitation;
    const color=displayValue === null ? null : colorFor(displayValue);
    const opacity=color ? state.highContrast ? Math.min(.94,.58 + displayValue / 36) : Math.min(.72,.22 + displayValue / 24) : 0;
    rect.setStyle({ fillColor: color || "transparent", fillOpacity: opacity });
  });
}
function distanceKm(lat1,lng1,lat2,lng2) { const x=(lng2-lng1)*111.32*Math.cos((lat1+lat2)*Math.PI/360), y=(lat2-lat1)*110.57; return Math.hypot(x,y); }
function pointInRing(lat,lng,ring) { let inside=false; for(let i=0,j=ring.length-1;i<ring.length;j=i++) { const [xi,yi]=ring[i], [xj,yj]=ring[j]; if(((yi>lat)!==(yj>lat)) && lng < (xj-xi)*(lat-yi)/(yj-yi)+xi) inside=!inside; } return inside; }
function pointInGeometry(lat,lng,geometry) {
  if (!geometry) return false;
  const polygons=geometry.type === "Polygon" ? [geometry.coordinates] : geometry.type === "MultiPolygon" ? geometry.coordinates : [];
  return polygons.some(polygon => pointInRing(lat,lng,polygon[0]) && !polygon.slice(1).some(hole => pointInRing(lat,lng,hole)));
}
function calculateMetrics() {
  if (state.weather.loading) return { loading:true, mode:timeMode() };
  if (state.weather.error) return { error:state.weather.error, mode:timeMode() };
  const nativeCells=state.weather.cells;
  const cells=nativeCells.filter(cell => state.mode === "coordinate"
    ? distanceKm(state.point.lat,state.point.lng,cell.lat,cell.lng)<=state.radiusKm
    : pointInGeometry(cell.lat,cell.lng,state.geometry)).map(cell=>weatherValue(cell,"precipitation"));
  if (!cells.length) cells.push(0);
  const point=nearestIncomingCell(state.point.lat,state.point.lng);
  const wet = cells.filter(v=>v>=state.coverageThreshold); const coverage=Math.round(wet.length/cells.length*100); const average=cells.reduce((a,b)=>a+b,0)/cells.length; const max=Math.max(...cells);
  const chance=weatherValue(point,"probability");
  // Open-Meteo precipitation is an hourly sum, therefore this is the selected hour's total.
  return { chance, coverage, average, max, accumulation: average, mode:timeMode() };
}
function renderMetrics() {
  const d=timeForIndex(state.index), metric=calculateMetrics(), mode=metric.mode;
  el.simulationDate.textContent = `${formatTime(d)} IST`;
  el.timelineTitle.textContent = `${formatTime(d, false)} IST${relativeDayLabel(d)}`;
  el.dataMode.textContent=mode; el.timelineMode.textContent="FORECAST MODEL";
  if(metric.loading || metric.error) { const label=metric.loading ? "Loading…" : "Unavailable"; el.chance.textContent=label; el.coverage.textContent=label; el.intensity.textContent=label; el.peak.textContent=label; el.accumulation.textContent=label; el.metricNote.textContent=metric.loading ? "Loading Open-Meteo forecast cells…" : `Open-Meteo forecast data could not be loaded: ${metric.error}`; updateRainLayer(); return; }
  el.chance.textContent=`${metric.chance}%`; el.coverage.textContent=`${metric.coverage}%`; el.intensity.textContent=`${metric.average.toFixed(1)} mm/hr`; el.peak.textContent=`${metric.max.toFixed(1)} mm/hr`; el.accumulation.textContent=`${metric.accumulation.toFixed(1)} mm`;
  el.liveStatus.innerHTML=state.playing?"<i></i> LIVE":"PAUSED"; el.metricNote.textContent=`Coverage uses Open-Meteo grid samples at or above ${state.coverageThreshold.toFixed(1)} mm/hr.`;
  updateRainLayer();
}
function render() { el.timeline.value=state.index; renderMetrics(); }
function chooseCoordinate(lat,lng,label="Selected coordinate",fit=true) {
  state.point={lat,lng,label}; state.bounds=null; state.geometry=null; state.approximateArea=false; state.regionKind="Coordinate"; state.mode="coordinate"; state.index=NOW_INDEX; activateTab("coordinate", false); el.modeBadge.textContent="Coordinates"; el.placeSearch.value=""; updateInputs(); updateSelection(fit); buildRainGrid(); loadOpenMeteoForecast();
}
function usePlace(name,lat,lng,geometry,bounds,approximateArea=false,regionKind="Area") { state.point={lat,lng,label:name}; state.bounds=bounds; state.geometry=geometry; state.approximateArea=approximateArea; state.regionKind=regionKind; state.mode="area"; state.index=NOW_INDEX; activateTab("area", false); el.modeBadge.textContent=approximateArea ? "Approximate PIN extent" : "Area boundary"; el.searchHint.hidden=!approximateArea; if(approximateArea) el.searchHint.textContent="This PIN code has no published boundary; the outlined extent is approximate."; el.searchResults.hidden=true; updateInputs(); updateSelection(true); buildRainGrid(); loadOpenMeteoForecast(); }
const stateCodes = { "Andhra Pradesh":"AP", "Arunachal Pradesh":"AR", Assam:"AS", Bihar:"BR", Chhattisgarh:"CG", Goa:"GA", Gujarat:"GJ", Haryana:"HR", "Himachal Pradesh":"HP", Jharkhand:"JH", Karnataka:"KA", Kerala:"KL", "Madhya Pradesh":"MP", Maharashtra:"MH", Manipur:"MN", Meghalaya:"ML", Mizoram:"MZ", Nagaland:"NL", Odisha:"OD", Punjab:"PB", Rajasthan:"RJ", Sikkim:"SK", "Tamil Nadu":"TN", Telangana:"TS", Tripura:"TR", "Uttar Pradesh":"UP", Uttarakhand:"UK", "West Bengal":"WB", Delhi:"DL", "Jammu and Kashmir":"JK", Ladakh:"LA", Puducherry:"PY", Chandigarh:"CH" };
function stateCode(result) { return stateCodes[result.address?.state] || result.address?.state || "India"; }
function resultName(result) { return result.name || result.display_name.split(",")[0]; }
function regionKind(result) {
  const type=result.addresstype || result.type || "region";
  const labels={ city:"City", town:"Town", village:"Village", state:"State", country:"Country", county:"District", district:"District", state_district:"District", city_district:"District", local_authority:"City", postal_code:"PIN code", postcode:"PIN code" };
  return labels[type] || type.replace(/_/g," ").replace(/\b\w/g,letter=>letter.toUpperCase());
}
function regionContext(result,kind) {
  const address=result.address || {};
  if (kind === "State") return address.country || "India";
  if (kind === "Country") return "";
  return address.state || address.country || "India";
}
function boundingBoxGeometry(bounds) { const [south,north,west,east]=bounds.map(Number); return { type:"Polygon", coordinates:[[[west,south],[east,south],[east,north],[west,north],[west,south]]] }; }
function selectSearchResult(result) {
  const [south,north,west,east]=result.boundingbox.map(Number);
  const hasBoundary=["Polygon","MultiPolygon"].includes(result.geojson?.type);
  const kind=regionKind(result), name=resultName(result), isTopLevelRegion=kind === "State" || kind === "Country" || result.address?.state === name, label=isTopLevelRegion ? name : `${name}, ${stateCode(result)}`;
  usePlace(label,Number(result.lat),Number(result.lon),hasBoundary ? result.geojson : boundingBoxGeometry(result.boundingbox),[[south,west],[north,east]],!hasBoundary,kind);
}
function renderSearchResults(results) {
  el.searchResults.replaceChildren();
  results.forEach(result => {
    if (!result.boundingbox && !result.lookupQuery) return;
    const button=document.createElement("button"); button.type="button"; button.className="search-result"; button.setAttribute("role","option");
    const kind=regionKind(result), context=regionContext(result,kind);
    button.innerHTML=`<strong>${resultName(result)}</strong><span>${kind}${context ? ` · ${context}` : ""}</span>`;
    button.addEventListener("click",()=>selectSearchResult(result));
    el.searchResults.append(button);
  });
  el.searchResults.hidden=el.searchResults.children.length===0;
}
function activateTab(tab, createPin=true) {
  const coordinate=tab === "coordinate";
  el.areaPanel.hidden=coordinate; el.coordinatePanel.hidden=!coordinate;
  el.areaTab.classList.toggle("is-active",!coordinate); el.coordinateTab.classList.toggle("is-active",coordinate);
  el.areaTab.setAttribute("aria-selected",String(!coordinate)); el.coordinateTab.setAttribute("aria-selected",String(coordinate));
  if (coordinate && createPin && state.mode !== "coordinate") chooseCoordinate(state.point.lat,state.point.lng,"Map pin",false);
}
async function findPlace() {
  const q=el.placeSearch.value.trim(); if(!q) return;
  const isPinCode=/^\d{6}$/.test(q);
  el.searchHint.hidden=true;
  el.searchButton.textContent="…"; el.searchButton.disabled=true;
  try {
    const res=await fetch(`https://nominatim.openstreetmap.org/search?format=jsonv2&addressdetails=1&polygon_geojson=1&limit=8&countrycodes=in&q=${encodeURIComponent(q)}`);
    const results=await res.json(); const filtered=results.filter(result => result.boundingbox && (["Polygon","MultiPolygon"].includes(result.geojson?.type) || (isPinCode && (result.type === "postal_code" || result.type === "postcode"))) && (!isPinCode || result.type === "postal_code" || result.type === "postcode"));
    if(!filtered.length) throw new Error();
    renderSearchResults(filtered);
  }
  catch { el.searchResults.hidden=true; el.searchHint.hidden=false; el.searchHint.textContent=isPinCode ? "No matching Indian PIN code was found." : "No matching region boundary was found. Try a more specific Indian name."; }
  finally { el.searchButton.textContent="Find"; el.searchButton.disabled=false; }
}
el.searchButton.addEventListener("click",findPlace); el.placeSearch.addEventListener("keydown",e=>{if(e.key==="Enter") findPlace();});
el.areaTab.addEventListener("click",()=>activateTab("area")); el.coordinateTab.addEventListener("click",()=>activateTab("coordinate"));
let coordinateInputTimer;
function applyTypedCoordinate() {
  const latitude=el.latitude.value.trim(), longitude=el.longitude.value.trim();
  if (!latitude || !longitude) return;
  const lat=Number(latitude), lng=Number(longitude);
  if(!Number.isFinite(lat) || !Number.isFinite(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) return;
  chooseCoordinate(lat,lng,"Manual coordinate");
}
[el.latitude,el.longitude].forEach(input=>input.addEventListener("input",()=>{ clearTimeout(coordinateInputTimer); coordinateInputTimer=setTimeout(applyTypedCoordinate,350); }));
el.advancedLink.addEventListener("click",()=>{ el.advancedLink.hidden=true; el.advancedPanel.hidden=false; });
el.advancedClose.addEventListener("click",()=>{ el.advancedPanel.hidden=true; el.advancedLink.hidden=false; });
el.coverageThreshold.addEventListener("input",()=>{ const value=Number(el.coverageThreshold.value); if(Number.isFinite(value) && value>=0) { state.coverageThreshold=value; renderMetrics(); } });
el.highContrast.addEventListener("change",()=>{ state.highContrast=el.highContrast.checked; updateRainLayer(); });
el.radius.addEventListener("change",()=>{ const v=Number(el.radius.value); if(!Number.isFinite(v)||v<.1){el.radius.value=state.radiusKm;return;} state.radiusKm=Math.round(v*10)/10; updateInputs(); updateSelection(false); buildRainGrid(); loadOpenMeteoForecast(); });
map.on("dblclick",e=>chooseCoordinate(e.latlng.lat,e.latlng.lng,"Selected coordinate"));
el.playButton.addEventListener("click",()=>{ state.playing=!state.playing; el.playButton.textContent=state.playing?"❚❚":"▶"; el.playButton.setAttribute("aria-label",state.playing?"Pause simulation":"Play simulation"); renderMetrics(); });
el.timeline.addEventListener("input",()=>{ state.index=Number(el.timeline.value); state.playing=false; el.playButton.textContent="▶"; render(); });
updateTimelineLabels();
map.on("mouseover",()=>el.clickPrompt.classList.add("visible")); map.on("mouseout",()=>el.clickPrompt.classList.remove("visible"));
updateInputs(); updateSelection(false); buildRainGrid(); loadOpenMeteoForecast();
async function loadInitialBoundary() {
  try {
    const response=await fetch("https://nominatim.openstreetmap.org/search?format=jsonv2&addressdetails=1&polygon_geojson=1&limit=3&countrycodes=in&q=Bengaluru");
    const results=await response.json(); const result=results.find(place => ["Polygon","MultiPolygon"].includes(place.geojson?.type));
    if (!result) return;
    const [south,north,west,east]=result.boundingbox.map(Number);
    usePlace(`${resultName(result)}, ${stateCode(result)}`,Number(result.lat),Number(result.lon),result.geojson,[[south,west],[north,east]]);
  } catch { /* The user can select an area manually if the geocoder is unavailable. */ }
}
loadInitialBoundary();
setInterval(()=>{ if(!state.playing)return; const last=(state.weather.times.length || windowTimes().length)-1; state.index=state.index>=last?0:state.index+1; render(); },850);
