Run PLANNED PR

Branch: pr-fix-cyrillic-02

You are working in the LAN_Transcriber repository. This PR fixes Cyrillic (Russian) text rendering. PR-UI-POLISH-02 added inter-cyrillic.woff2 and the @font-face declaration, but Cyrillic text still renders in a visually different font compared to Latin characters. Follow AGENTS.md exactly.

This task must be executed in 3 internal phases within a single run.

Phase 1 - Diagnose
Investigate why Cyrillic text still looks different despite the font file and @font-face being present:
1. Verify inter-cyrillic.woff2 is a real Inter font file (check file size, should be around 73KB, and validate it is woff2 format)
2. Check the Tailwind CDN script (tailwind.js) - when Tailwind runs in the browser, it generates CSS for the font-display utility class. Verify that the generated font-family includes Inter. The current Tailwind config has fontFamily.display set to ["Inter"] with no fallbacks. Tailwind may generate font-family: Inter (without quotes) which could fail to match the @font-face declaration that uses font-family: 'Inter' (with quotes). Test both variants.
3. Check if the body CSS rule font-family: 'Inter', sans-serif conflicts with or is overridden by the Tailwind font-display class
4. Check if any element uses a Tailwind class like font-sans that overrides font-display
5. Verify that the woff2 file is actually being served by the FastAPI static files mount (check /static/fonts/inter-cyrillic.woff2 is accessible)

Phase 2 - Fix
Based on the diagnosis, apply the necessary fixes. The most likely issues are:

FIX A: Tailwind fontFamily fallback stack
Change the Tailwind config in base.html from:
  fontFamily: { "display": ["Inter"] }
to:
  fontFamily: { "display": ["Inter", "system-ui", "-apple-system", "sans-serif"], "sans": ["Inter", "system-ui", "-apple-system", "sans-serif"] }
This ensures that both font-display and font-sans utility classes resolve to Inter with proper fallbacks. The fallback stack ensures Cyrillic glyphs not covered by the woff2 subset still render in a visually similar system font.

FIX B: Ensure @font-face uses unquoted family name to match Tailwind output
Tailwind CDN generates font-family: Inter (no quotes). The @font-face blocks in base.html use font-family: 'Inter' (with quotes). According to CSS spec, both should match, but some browsers may be inconsistent. To be safe, add the unquoted variant or ensure consistency. Test with the browser DevTools network tab to verify the woff2 files are actually loaded.

FIX C: If the woff2 file is not actually Inter
If investigation reveals the downloaded woff2 is corrupt or not Inter (the curl URL in PR-UI-POLISH-02 may have downloaded an error page or redirect), re-download it:
  curl -L -o lan_app/static/fonts/inter-cyrillic.woff2 "https://fonts.gstatic.com/s/inter/v18/UcCo3FwrK3iLTcviYwY.woff2"
If the network is restricted, generate a Cyrillic subset from the full Inter variable font, or switch the approach: use the Google Fonts CSS API to serve Inter with Cyrillic support via a link tag as fallback:
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400..900&display=swap&subset=cyrillic" rel="stylesheet">
If external network is not available at runtime either, ensure the fallback font stack (system-ui, Segoe UI) provides acceptable Cyrillic rendering and document that Inter Cyrillic requires the woff2 file to be manually placed.

Hard constraints
- Do not change colors, spacing, or layout
- Do not modify Python files except test files
- Do not add new features
- Only fix font rendering

Phase 3 - Verify
- Run scripts/ci.sh until exit code 0
- Generate required review artifacts per AGENTS.md
- Verify in a browser that Cyrillic text on the recording details page, summary, and transcript tabs renders in Inter (or a visually consistent fallback)

Success criteria
- Cyrillic characters render in the same visual weight and style as Latin characters across all pages
- The @font-face declaration and Tailwind fontFamily config are consistent
- The inter-cyrillic.woff2 file is confirmed to be a valid Inter font subset
- No visual difference between Russian and English text in the UI
- CI passes
