# "Permanent Way" — design specification

The complete visual system RAILPULSE uses. Every value below is copied from
the live stylesheet and checked against it, so this can be replicated without
reading the dashboard source.

**Contents** — [Concept](#1-concept) · [Fonts](#2-fonts) ·
[Type scale](#3-type-scale) · [Colour](#4-colour) ·
[Spacing & radius](#5-spacing-and-radius) · [Layout](#6-layout) ·
[Components](#7-components) · [Interaction states](#8-interaction-states) ·
[Motion](#9-motion) · [Theming](#10-theming) · [Print](#11-print) ·
[Rules](#12-rules) · [Prompt version](#13-prompt-version)

---

## 1. Concept

A railway engineering document, not a SaaS dashboard.

Four decisions carry the whole thing, and if you only take four, take these:

1. **The page is darker than the panels.** `--bg` is a mid warm grey; panels
   are near-white. Panes read as sheets laid on a desk rather than cards
   floating over a canvas. This single inversion does more to break the
   template look than anything else here.
2. **No drop shadows on surfaces.** Depth is a 1px hairline plus a change of
   ground. `--shadow-sm` and `--shadow-md` are literally `none`.
3. **The accent is deliberately not blue.** Petrol teal — an enamel-and-
   machinery colour. Blue, indigo and violet are the tell.
4. **Every number is monospace with tabular figures.** Readouts look like
   instruments, and a value updating on a live poll cannot make a row jitter.

Underneath that: status colours are **railway signal aspects** (green,
yellow, double yellow, red), which is the exact semantics a fault dashboard
needs and is readable without a legend. Typography is transit signage —
condensed grotesk headings, regular grotesk prose, true mono for data.

---

## 2. Fonts

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
```

| Token | Family | Weights | Used for |
|-------|--------|---------|----------|
| `--font-heading` | **Barlow Condensed** | 500 · 600 · 700 | Panel titles, wordmark, badges, SVG diagram labels |
| `--font-body` | **Barlow** | 400 · 500 · 600 · 700 | All prose, labels, tabs, buttons, inputs |
| `--font-mono` | **JetBrains Mono** | 400 · 500 · 700 | Every figure, unit ID, timestamp, table number |

Barlow was drawn from the grotesques used on public-transit signage. That is
why it is here instead of Inter — it carries the domain, and it is narrower,
which is why the body size is 15px rather than 14px.

Weights in use: `400, 500, 600, 650, 700`. Nothing heavier; the condensed
face at 600 already reads as emphatic.

---

## 3. Type scale

| px | Weight | Family | Applied to |
|----|--------|--------|------------|
| 32 | 700 | Condensed | Wordmark (uppercase, +1.5px tracking) |
| 26 | 700 | Mono | Health Index headline figure |
| 24 | 700 | Mono | Stat tile numbers |
| 23 | 700 | Mono | Business-impact headline figures |
| 21 | 600 | Mono | Unit detail title |
| 19 | 600 | Condensed | Panel headings (uppercase, +1.1px, rule under) |
| 17 | 600 | Mono | Inline emphasis figures |
| 15 | 400 | Barlow | **Body default**, line-height 1.55 |
| 13.5 | 600 | Barlow | Tabs, primary buttons |
| 12.5 | 400 | Barlow | Table cells, dense prose |
| 12 | 600 | Condensed | Badges (uppercase, +0.9px) |
| 11.5 | 600 | Barlow | Secondary buttons, filter toggles |
| 10.5 | 400 | Barlow | Captions, sub-notes, footnotes |
| 10 | 600 | Barlow | Table headers, stat labels (uppercase, +0.4–0.5px) |

**Tracking ladder.** Negative for figures, positive for anything uppercase:
`-0.5px` mono headline figures · `-0.2px` inline mono · `+0.2px` tabs ·
`+0.4px` table headers · `+0.9px` badges · `+1.1px` panel headings ·
`+1.5px` wordmark. The rule is simple — the bigger and the more uppercase,
the more tracking.

---

## 4. Colour

### Light (default)

```css
:root {
  color-scheme: light;

  --bg: #E7E2D6;              /* page — DARKER than the panels */
  --panel: #F5F2EA;           /* card / sheet */
  --panel-alt: #EDE8DC;       /* inset, tile, secondary surface */
  --border-rgb: 199, 190, 172;
  --border: rgb(var(--border-rgb));
  --border-strong: #A79C84;   /* hover edge only */

  --text: #17150F;            /* graphite, warm — not blue-black */
  --text-dim: #5E5748;

  --accent: #13565C;          /* petrol teal */
  --accent-rgb: 19, 86, 92;
  --accent-contrast: #F5F2EA; /* text ON an accent fill */

  --normal:   #1D6B3F;  --normal-rgb:   29, 107, 63;    /* signal green  */
  --caution:  #8A6413;  --caution-rgb:  138, 100, 19;   /* signal yellow */
  --warning:  #A2500F;  --warning-rgb:  162, 80, 15;    /* double yellow */
  --critical: #A81F24;  --critical-rgb: 168, 31, 36;    /* signal red    */

  --shadow-sm: none;
  --shadow-md: none;
  --shadow-lg: 0 10px 30px rgba(23, 21, 15, 0.13);   /* overlays only */
}
```

### Dark

Not an inversion — a re-tune. Ground goes warm near-black rather than
blue-black, and the aspects brighten to hold contrast on a dark card.

```css
[data-theme="dark"] {
  color-scheme: dark;

  --bg: #111010;
  --panel: #1A1814;
  --panel-alt: #221F1A;
  --border-rgb: 58, 53, 44;
  --border: rgb(var(--border-rgb));
  --border-strong: #574F42;

  --text: #EDE8DC;
  --text-dim: #9C9384;

  --accent: #5FB8C0;
  --accent-rgb: 95, 184, 192;
  --accent-contrast: #111010;

  --normal:   #49C97C;  --normal-rgb:   73, 201, 124;
  --caution:  #E0B341;  --caution-rgb:  224, 179, 65;
  --warning:  #F0873B;  --warning-rgb:  240, 135, 59;
  --critical: #FF5F5F;  --critical-rgb: 255, 95, 95;

  --shadow-sm: none;
  --shadow-md: none;
  --shadow-lg: 0 10px 32px rgba(0, 0, 0, 0.5);
}
```

### Measured contrast

Each foreground against **its own theme's card**, not against white:

| Token | Light on `#F5F2EA` | Dark on `#1A1814` |
|-------|-----------------:|----------------:|
| `--text` | 16.31:1 | 14.50:1 |
| `--text-dim` | 6.40:1 | 5.84:1 |
| `--accent` | 7.47:1 | 7.68:1 |
| `--normal` | 5.82:1 | 8.38:1 |
| `--caution` | 4.80:1 | 9.03:1 |
| `--warning` | 5.09:1 | 6.95:1 |
| `--critical` | 6.50:1 | 5.95:1 |

All clear WCAG AA. The four aspects sit close in **lightness**, so they are
separable by hue but not in greyscale — therefore colour never carries
meaning alone. Every badge, tile and row states the level in words too.

### Tint convention

Status surfaces use the `-rgb` triplet at fixed alphas, never a second hex:

- `rgba(var(--x-rgb), 0.12)` — badge fill
- `rgba(var(--x-rgb), 0.4)` — card border at rest
- `rgba(var(--accent-rgb), 0.14)` — selected table row
- `rgba(var(--accent-rgb), 0.16)` — active filter button
- `rgba(var(--accent-rgb), 0.07)` — drop-zone hover

---

## 5. Spacing and radius

```css
--sp-1: 8px;  --sp-2: 16px;  --sp-3: 24px;  --sp-4: 32px;
--radius-lg: 3px;  --radius-md: 2px;  --radius-sm: 2px;  --radius-pill: 2px;
```

An 8px base scale. `--radius-pill` is deliberately **2px, not 999px** — the
token is kept so existing pill call-sites stay valid while rendering square.
Nothing on the page is round except the chat launcher, which sets `50%`
directly.

Panel padding is `--sp-3` (24px). Grid gaps are `--sp-3`. Internal component
padding is hand-set in the 6–16px range and does not follow the scale — the
scale governs layout, not component interiors.

---

## 6. Layout

```css
main {
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: var(--sp-3);
  padding: var(--sp-3);
  max-width: 1440px;
  margin: 0 auto;
}
.full-width-panel { grid-column: 1 / -1; }

.twin-detail-row {          /* nested two-column row */
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: var(--sp-3);
}
```

A 12-ish column feel achieved with `1fr + 380px`: a wide working column and a
fixed detail rail. 380px is the recurring rail width — use it anywhere a
sidebar appears so the vertical rhythm lines up between rows.

### Breakpoints

| Width | Change |
|-------|--------|
| `1100px` | Business-impact headline drops from 4 to 2 columns |
| `1000px` | Impact form/results stack to one column |
| `900px` | `main` and `.twin-detail-row` collapse to `1fr`; service cards stack |
| `760px` | Role tabs scroll horizontally, subtitles hidden |
| `560px` | Chat dock goes edge-to-edge with 8px insets |

---

## 7. Components

### Page frame

```css
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  font-size: 15px;
  line-height: 1.55;
}

header {
  padding: var(--sp-2) 0;
  border-bottom: 1px solid var(--border);
  background: var(--bg);
}

.wordmark {
  font-family: var(--font-heading);
  font-size: 32px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--accent);
  line-height: 1.1;
  margin: 0;
}
.brand-logo { height: 30px; width: auto; display: block; }
```

### Panel — the core surface

```css
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: var(--sp-3);
  box-shadow: var(--shadow-md);   /* = none */
}

/* Section head on a printed form: a rule across the whole pane. */
.panel h2 {
  display: flex;
  align-items: center;
  gap: 9px;
  font-family: var(--font-heading);
  font-size: 19px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.1px;
  color: var(--text);
  margin: 0 0 var(--sp-2) 0;
  padding-bottom: 9px;
  border-bottom: 1px solid var(--border);
}
```

Each heading carries a 16×16 stroked SVG icon at `stroke-width: 1.8`,
`currentColor`, no fill. Line icons only — never filled, never emoji.

### Stat tiles

```css
.stat-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: var(--sp-2); margin-bottom: var(--sp-2); }

.stat-tile {
  background: var(--panel-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 10px;
  text-align: center;
  transition: border-color 0.15s ease, transform 0.15s ease;
}
.stat-tile:hover { border-color: var(--border-strong); transform: translateY(-1px); }

.stat-tile .num {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.5px;
  font-size: 24px;
  font-weight: 700;
  line-height: 1.1;
  color: var(--text);
}
```

### Badge — a signage label, not a pill

```css
.badge {
  display: inline-block;
  padding: 3px 9px 3px 8px;
  border-radius: var(--radius-sm);
  font-family: var(--font-heading);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.9px;
  border-left-width: 3px;
  border-left-style: solid;
}
.badge.Normal   { background: rgba(var(--normal-rgb), 0.12);   color: var(--normal);   border-color: var(--normal); }
.badge.Caution  { background: rgba(var(--caution-rgb), 0.12);  color: var(--caution);  border-color: var(--caution); }
.badge.Warning  { background: rgba(var(--warning-rgb), 0.12);  color: var(--warning);  border-color: var(--warning); }
.badge.Critical { background: rgba(var(--critical-rgb), 0.12); color: var(--critical); border-color: var(--critical); }
```

The 3px aspect bar down the leading edge is the signature — it reads as a
trackside marker rather than a UI chip.

### Alert card

```css
.priority-card {
  background: var(--panel-alt);
  border: 1px solid rgba(var(--critical-rgb), 0.4);
  border-left: 3px solid var(--critical);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  cursor: pointer;
  transition: border-color 0.15s ease, transform 0.15s ease;
}
.priority-card:hover { border-color: var(--critical); transform: translateY(-1px); }
```

Same 3px leading bar as the badge. It is the system's one repeated motif.

### Tables

```css
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }

th {
  color: var(--text-dim);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  font-size: 10.5px;
}

tr.unit-row:hover    { background: var(--panel-alt); }
tr.unit-row.selected { background: rgba(var(--accent-rgb), 0.14); }
```

Rows are separated by a 1px bottom hairline, never by zebra striping. Numeric
columns get `.mono` and right-align.

### Numbers

```css
.mono {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum" 1;
  font-size: 0.94em;        /* mono runs large; pull it back optically */
  letter-spacing: -0.2px;
}
.num-col { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 0.94em; }
```

### Tabs — underline, not segmented

```css
.role-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); }

.role-tab {
  appearance: none;
  border: 0;
  border-bottom: 2px solid transparent;
  background: transparent;
  cursor: pointer;
  font-family: var(--font-body);
  font-size: 13.5px;
  font-weight: 600;
  letter-spacing: 0.2px;
  color: var(--text-dim);
  padding: 13px 22px 11px;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: color 0.18s ease, border-color 0.18s ease;
}
.role-tab:hover { color: var(--text); }
.role-tab:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.role-tab[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }
```

Each tab carries a second line — a 10.5px/400 subtitle saying what the view
is for. Hidden below 760px.

### Numeric input

```css
.impact-field input {
  width: 100%;
  background: var(--panel-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 7px 9px;
  color: var(--text);
  font-size: 13px;
  font-weight: 600;
  font-family: var(--font-body);
  font-variant-numeric: tabular-nums;
  text-align: right;                 /* money and counts right-align */
  transition: border-color 0.18s ease;
}
.impact-field input:focus { outline: none; border-color: var(--accent); }
.impact-field input::-webkit-outer-spin-button,
.impact-field input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.impact-field input[type=number] { -moz-appearance: textfield; }
```

Units (`$`, `h`, `%`, `×`) are absolutely-positioned `span`s inside the
field, not part of the label, with matching input padding. Spinners removed —
they are noise on a dense form.

Form rows are `label` (flex 1, 11.5px, dim) + a fixed 104px field, separated
by `1px solid rgba(var(--accent-rgb), 0.07)`. Groups get a 10px/700 uppercase
accent-coloured head with a full border under it.

### Drop zone

```css
.drop-zone {
  border: 1.5px dashed rgba(var(--border-rgb), 0.9);
  border-radius: var(--radius-lg);
  padding: var(--sp-4) var(--sp-3);
  text-align: center;
  background: var(--panel-alt);
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}
.drop-zone:hover, .drop-zone.dragover {
  border-color: var(--accent);
  background: rgba(var(--accent-rgb), 0.07);
}
```

### Glossary tooltip

```css
.gloss { border-bottom: 1px dotted var(--text-dim); cursor: help; position: relative; outline: none; }
.gloss::after {
  content: attr(data-tip);
  position: absolute;
  left: 0;
  bottom: calc(100% + 7px);
  z-index: 60;
  width: max-content;
  max-width: 280px;
  padding: 9px 11px;
  border-radius: var(--radius-md);
  background: var(--bg);
  border: 1px solid var(--accent);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
  font-size: 11.5px;
  font-weight: 400;
  line-height: 1.5;
  color: var(--text);
  text-transform: none;
  letter-spacing: 0;
  white-space: normal;
  text-align: left;
  opacity: 0;
  visibility: hidden;
  transform: translateY(3px);
  transition: opacity 0.15s ease, transform 0.15s ease, visibility 0.15s ease;
}
.gloss:hover::after, .gloss:focus::after { opacity: 1; visibility: visible; transform: translateY(0); }
```

CSS-only, `tabindex="0"` so it opens on keyboard focus too. No library, no
layout shift.

### Theme toggle

```css
.theme-toggle {
  width: 36px;
  height: 36px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text-dim);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: color 0.18s ease, border-color 0.18s ease;
}
.theme-toggle:hover { color: var(--accent); border-color: var(--accent); }
.theme-toggle .icon-sun  { display: none; }
.theme-toggle .icon-moon { display: block; }
[data-theme="dark"] .theme-toggle .icon-sun  { display: block; }
[data-theme="dark"] .theme-toggle .icon-moon { display: none; }
```

Shows the theme you would switch **to** — a moon means "go dark".

### Floating dock

```css
.chat-dock {
  position: fixed;
  right: 24px;
  bottom: 24px;
  z-index: 95;
  width: min(430px, calc(100vw - 32px));
  max-height: min(660px, calc(100vh - 48px));
  display: none;
  flex-direction: column;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--panel);
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.55);
  overflow: hidden;
}
.chat-dock.open { display: flex; }
```

430px keeps a line of prose near 60 characters. Overlays are the **only**
place a shadow is allowed — they genuinely float, panels do not.

### Diagram grid texture

CSS, not an image, so it follows the theme and scales:

```css
background-image:
  linear-gradient(to right,  rgba(var(--border-rgb), 0.55) 1px, transparent 1px),
  linear-gradient(to bottom, rgba(var(--border-rgb), 0.55) 1px, transparent 1px);
background-size: 22px 22px;
```

---

## 8. Interaction states

One convention, applied everywhere:

| State | Treatment |
|-------|-----------|
| **Hover, surface** | `border-color: var(--border-strong)` + `transform: translateY(-1px)` |
| **Hover, control** | `border-color` and `color` both → `var(--accent)` |
| **Hover, table row** | `background: var(--panel-alt)` |
| **Selected, row** | `background: rgba(var(--accent-rgb), 0.14)` |
| **Active, toggle** | `background: rgba(var(--accent-rgb), 0.16)` + `color: var(--accent)` |
| **Focus** | `outline: 2px solid var(--accent)` — **never removed** |
| **Selected, SVG** | `stroke: var(--accent); stroke-width: 2.5px` |
| **Hover, SVG** | `filter: brightness(1.3); stroke: var(--bg); stroke-width: 2px` |

Note the SVG hover halo is `--bg`, not white — white vanishes on a light
card.

---

## 9. Motion

Durations are 0.12s–0.18s, easing is always `ease`. Nothing is longer, and
nothing bounces.

| Duration | Used for |
|----------|----------|
| `0.12s` | Background-only changes |
| `0.15s` | Surface hover, tooltips, SVG fill/stroke |
| `0.18s` | Controls, tabs, theme toggle, dock |

Only `transform`, `opacity`, `color`, `background`, `border-color` and
`filter` are animated — never width, height or layout properties.

```css
@media (prefers-reduced-motion: reduce) {
  .chat-typing span { animation: none; opacity: 0.55; }
  .chat-fab, .role-tab, .wc-btn, .gloss::after { transition: none; }
  .copilot-spinner, .upload-spinner { animation: none; }
}
```

---

## 10. Theming

Light is default. Dark applies via `data-theme="dark"` on `<html>`.

**Resolve before first paint**, in `<head>` — otherwise dark-mode visitors
get a white flash:

```html
<script>
  (function () {
    try {
      var saved = localStorage.getItem("app.theme");
      var dark = saved ? saved === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      if (dark) document.documentElement.setAttribute("data-theme", "dark");
    } catch (e) {}   /* private mode → light default */
  })();
</script>
```

Follow the OS until the visitor picks a theme; once they do, their choice
wins and the OS listener stops applying.

**Canvas and hand-built SVG cannot inherit custom properties.** Read them
back and re-render on every theme change:

```js
function readThemeColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = n => cs.getPropertyValue(n).trim();
  RISK_COLORS = { Normal: v("--normal"), Caution: v("--caution"),
                  Warning: v("--warning"), Critical: v("--critical") };
  TEXT_DIM_HEX = v("--text-dim");
  BORDER_HEX   = v("--border-strong");
  BG_HEX       = v("--panel");   /* SVG sits on the card, not the page */
  CHART_GRID   = `rgba(${v("--border-rgb")}, 0.75)`;
}
```

Declare those mirrors as `let`, never `const`, and never cache them into a
further constant — a stale copy is the classic half-themed bug.

---

## 11. Print

The screen UI is suppressed entirely and one document is revealed:

```css
@media print {
  body { background: #fff; color: #000; }
  body > *, .chat-dock, .chat-fab { display: none !important; }
  #workCard { display: block !important; color: #000; font-family: Georgia, "Times New Roman", serif; }
  #workCard table { width: 100%; border-collapse: collapse; font-size: 12px; }
  #workCard th, #workCard td { border: 1px solid #999; padding: 5px 7px; text-align: left; }
}
```

Print switches to a **serif** and to pure black on white. Screen and paper
are different media and should not pretend otherwise.

---

## 12. Rules

**Do**

- Keep the page darker than the panels.
- Put every colour in a token. No hex in markup, JS or chart options.
- Set every number in the mono with tabular figures.
- Rule under every panel heading.
- State status in words as well as colour.
- Keep focus outlines. 2px, accent, never removed.
- Animate only transform / opacity / colour, 0.12–0.18s, `ease`.

**Don't**

- No drop shadows on panels or tiles. Overlays only.
- No radius above 3px. No pills.
- No cool greys. Every neutral carries a yellow-red bias.
- No blue, indigo or violet accent.
- No status colour for anything non-status — signal red means Critical.
- No second display family. One condensed, one grotesk, one mono.
- No zebra striping; hairlines separate rows.
- No filled icons, no emoji. Stroked line icons at 1.8.

---

## 13. Prompt version

> Build it in the "Permanent Way" style: a railway engineering document, not
> a SaaS dashboard. Warm drafting-paper ground `#E7E2D6` with panels
> **lighter** than the page at `#F5F2EA`, insets `#EDE8DC`, graphite ink
> `#17150F`, dim ink `#5E5748`, hairline borders `#C7BEAC`, and **no drop
> shadows on surfaces** — depth comes from a 1px rule and a shift of ground;
> shadows are allowed only on floating overlays. Radius 2–3px everywhere,
> never pills. Accent is petrol teal `#13565C`, never blue or indigo. Status
> colours are railway signal aspects used for status only: green `#1D6B3F`,
> yellow `#8A6413`, double yellow `#A2500F`, red `#A81F24`; tint their
> surfaces with `rgba(r,g,b,0.12)` rather than a second hex. Type: Barlow
> Condensed 600 uppercase with +1.1px tracking for panel headings, each with
> a rule underneath; Barlow 400 at 15px/1.55 for prose; JetBrains Mono with
> tabular figures at 0.94em for **every** number, so readouts look like
> instruments. Badges are square signage labels with a 3px status bar down
> the left edge — reuse that bar on alert cards as the one repeated motif.
> Tabs are underline tabs with a 2px accent border-bottom, not segmented
> pills. Tables use hairline row separators, never zebra striping. Hover
> lifts a surface 1px and strengthens its border; controls turn accent on
> hover; focus is a 2px accent outline, never removed. Motion is 0.12–0.18s
> `ease` on transform/opacity/colour only, disabled under
> `prefers-reduced-motion`. Layout is a `1fr 380px` grid, 24px gutters,
> 1440px max width, collapsing to one column at 900px. Dark mode via
> `data-theme="dark"` inverts to warm near-black `#111010`, panels `#1A1814`,
> insets `#221F1A`, accent `#5FB8C0` and brightened aspects — warm
> throughout, never blue-black — resolved in a `<head>` script before first
> paint. Put every colour in a CSS custom property and let nothing else hold
> a hex.
