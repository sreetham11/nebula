# "Permanent Way" — design specification

The exact visual system RAILPULSE uses. Everything below is copied from the
live stylesheet, not described from memory — paste it and you get the same
result.

---

## The idea in one paragraph

A railway engineering document, not a SaaS dashboard. Warm drafting paper
instead of cool grey, railway **signal aspects** (green / yellow / double
yellow / red) as the status colours, and depth built from hairlines and a
shift of ground rather than drop shadows. Corners are effectively square.
Typography is transit signage: condensed grotesk headings, regular grotesk
prose, and a true monospace for every number so readouts look like
instruments. The interaction accent is petrol teal — an enamel-and-machinery
colour deliberately chosen to sit nowhere near the status ramp, and nowhere
near the blue every generated dashboard reaches for.

---

## 1. Fonts

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
```

| Role | Family | Used for |
|------|--------|----------|
| Heading | **Barlow Condensed** | Panel titles, wordmark, badges, SVG labels |
| Body | **Barlow** | All prose, labels, tabs, buttons |
| Mono | **JetBrains Mono** | Every figure, unit ID, timestamp |

Barlow was drawn from the grotesques used on public-transit signage — that is
why it is here rather than Inter.

## 2. Tokens

```css
:root {
  color-scheme: light;

  --bg: #E7E2D6;              /* page — DARKER than the panels */
  --panel: #F5F2EA;           /* card */
  --panel-alt: #EDE8DC;       /* inset / secondary surface */
  --border-rgb: 199, 190, 172;
  --border: rgb(var(--border-rgb));
  --border-strong: #A79C84;

  --text: #17150F;
  --text-dim: #5E5748;

  --accent: #13565C;          /* petrol teal */
  --accent-rgb: 19, 86, 92;
  --accent-contrast: #F5F2EA; /* text ON the accent fill */

  --normal:   #1D6B3F;  --normal-rgb:   29, 107, 63;    /* signal green   */
  --caution:  #8A6413;  --caution-rgb:  138, 100, 19;   /* signal yellow  */
  --warning:  #A2500F;  --warning-rgb:  162, 80, 15;    /* double yellow  */
  --critical: #A81F24;  --critical-rgb: 168, 31, 36;    /* signal red     */

  --shadow-sm: none;
  --shadow-md: none;
  --shadow-lg: 0 10px 30px rgba(23, 21, 15, 0.13);   /* modals only */

  --font-heading: "Barlow Condensed", "Barlow", Arial Narrow, sans-serif;
  --font-body: "Barlow", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;

  --sp-1: 8px;  --sp-2: 16px;  --sp-3: 24px;  --sp-4: 32px;

  --radius-lg: 3px;  --radius-md: 2px;  --radius-sm: 2px;  --radius-pill: 2px;
}

[data-theme="dark"] {
  color-scheme: dark;

  --bg: #111010;              /* warm near-black, NOT blue-black */
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

**Contrast, measured against each theme's own card:** light 4.8–16.3:1, dark
5.8–14.5:1. All AA. The four aspects sit close in lightness, so colour never
carries meaning alone — every badge and row states the level in words too.

## 3. Components

```css
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  font-size: 15px;          /* Barlow is narrow; 14px reads small */
  line-height: 1.55;
}

main {
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: var(--sp-3);
  padding: var(--sp-3);
  max-width: 1440px;
  margin: 0 auto;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: var(--sp-3);
  box-shadow: var(--shadow-md);   /* = none in both themes */
}

/* Section head on a printed form: rule across the pane. */
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

/* Signage label, not a pill: square, tracked, aspect bar on the edge. */
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

/* Every figure. Tabular so a live-polling value cannot make a row jitter. */
.mono {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum" 1;
  font-size: 0.94em;
  letter-spacing: -0.2px;
}

/* Headline readouts look like instruments. */
.stat-tile .num {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.5px;
  font-size: 24px;
  font-weight: 700;
  line-height: 1.1;
  color: var(--text);
}

/* Underline tabs — navigation, not a segmented filter. */
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
  transition: color 0.18s ease, border-color 0.18s ease;
}
.role-tab[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent); }

/* Jargon carries its definition at the point of use. */
.gloss { border-bottom: 1px dotted var(--text-dim); cursor: help; position: relative; }
```

**Grid texture** (behind the diagram panel) is CSS, not an image, so it
follows the theme:

```css
background-image:
  linear-gradient(to right,  rgba(var(--border-rgb), 0.55) 1px, transparent 1px),
  linear-gradient(to bottom, rgba(var(--border-rgb), 0.55) 1px, transparent 1px);
background-size: 22px 22px;
```

## 4. Theme switching

Light is the default; dark applies via `data-theme="dark"` on `<html>`.
Resolve it **before first paint** or dark-mode visitors get a white flash:

```html
<script>
  (function () {
    try {
      var saved = localStorage.getItem("app.theme");
      var dark = saved ? saved === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      if (dark) document.documentElement.setAttribute("data-theme", "dark");
    } catch (e) {}
  })();
</script>
```

Anything painting to a `<canvas>` or a hand-built SVG string cannot inherit
CSS variables — read them back and re-render on every theme change:

```js
const cs = getComputedStyle(document.documentElement);
const v = n => cs.getPropertyValue(n).trim();
```

## 5. Rules

**Do**

- Keep the page darker than the panels. Panes read as sheets on a desk.
- Put every colour in a token. No hex anywhere else — not in markup, not in
  JS, not in chart options.
- Set every number in the mono with tabular figures.
- Give each panel heading a rule across the pane.
- State status in words as well as colour.

**Don't**

- No drop shadows on cards. Depth is hairlines and a change of ground.
- No rounded corners beyond 3px. No pills.
- No cool greys. Every neutral carries a yellow-red bias.
- No blue, indigo or violet accent. That is the tell.
- Don't use a status colour for anything non-status — signal red means
  Critical and nothing else.
- Don't mix a second display family. One condensed grotesk, one grotesk, one
  mono.

---

## Prompt version

> Build it in the "Permanent Way" style: a railway engineering document, not a
> SaaS dashboard. Warm drafting-paper ground `#E7E2D6` with panels **lighter**
> than the page at `#F5F2EA`, graphite ink `#17150F`, hairline borders
> `#C7BEAC`, and **no drop shadows** — depth comes from rules and a shift of
> ground. Corners 2–3px, never pills. Accent is petrol teal `#13565C`; never
> blue or indigo. Status colours are railway signal aspects: green `#1D6B3F`,
> yellow `#8A6413`, double yellow `#A2500F`, red `#A81F24` — used for status
> only. Type is Barlow Condensed for headings (uppercase, ~1.1px tracking,
> with a rule under each panel title), Barlow for prose at 15px/1.55, and
> JetBrains Mono with tabular figures for every number, so readouts look like
> instruments. Badges are square signage labels with a 3px aspect bar down the
> left edge, not tinted pills. Dark mode inverts to warm near-black `#111010`
> panels `#1A1814` with brightened aspects — warm, never blue-black. Put every
> colour in a CSS custom property and let nothing else hold a hex.
