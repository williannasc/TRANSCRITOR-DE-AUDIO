import React, { useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const defaultConfig = {
  fontFamily: 'Arial',
  baseFontSize: 14,
  titleColor: '#d9531e',
  artistColor: '#7cb342',
  lyricColor: '#222222',
  columns: 2,
};

const parseStanzas = (text) => {
  const normalized = (text || '').replace(/\r/g, '').trim();

  if (!normalized) {
    return [['']];
  }

  return normalized
    .split(/\n\s*\n+/)
    .map((stanza) => stanza.split('\n').map((line) => line.trim()).filter(Boolean));
};

const escapeHtml = (value) =>
  String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');

function LyricsPrintPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const payload = location.state || {};

  const title = payload.title || 'Barquinho - Ao Vivo';
  const artist = payload.artist || 'Tradição';
  const transcript = payload.transcript || 'Cole a letra aqui...';
  const compositor = payload.compositor || artist;

  const [fontFamily, setFontFamily] = useState(defaultConfig.fontFamily);
  const [baseFontSize, setBaseFontSize] = useState(defaultConfig.baseFontSize);
  const [titleColor, setTitleColor] = useState(defaultConfig.titleColor);
  const [artistColor, setArtistColor] = useState(defaultConfig.artistColor);
  const [lyricColor, setLyricColor] = useState(defaultConfig.lyricColor);
  const [columns, setColumns] = useState(defaultConfig.columns);

  const stanzas = useMemo(() => parseStanzas(transcript), [transcript]);

  const resetDefaults = () => {
    setFontFamily(defaultConfig.fontFamily);
    setBaseFontSize(defaultConfig.baseFontSize);
    setTitleColor(defaultConfig.titleColor);
    setArtistColor(defaultConfig.artistColor);
    setLyricColor(defaultConfig.lyricColor);
    setColumns(defaultConfig.columns);
  };

  const handleImprimir = () => {
    const iframeAntigo = document.getElementById('print-iframe-worker');
    if (iframeAntigo) {
      iframeAntigo.remove();
    }

    const printIframe = document.createElement('iframe');
    printIframe.id = 'print-iframe-worker';
    printIframe.style.position = 'fixed';
    printIframe.style.right = '0';
    printIframe.style.bottom = '0';
    printIframe.style.width = '0';
    printIframe.style.height = '0';
    printIframe.style.border = '0';
    document.body.appendChild(printIframe);

    const doc = printIframe.contentWindow?.document || printIframe.contentDocument;
    if (!doc) return;

    const tamanhoTitulo = Math.round(baseFontSize * 1.8);
    const tamanhoArtista = Math.round(baseFontSize * 1.25);
    const espacamentoEstrofe = Math.round(baseFontSize * 1.2);

    doc.open();
    doc.write(`
      <!DOCTYPE html>
      <html lang="pt-BR">
        <head>
          <meta charset="UTF-8" />
          <title>${escapeHtml(title)} - ${escapeHtml(artist)}</title>
          <style>
            @page {
              size: A4 portrait;
              margin: 12mm 15mm;
            }
            * {
              box-sizing: border-box;
              -webkit-print-color-adjust: exact !important;
              print-color-adjust: exact !important;
            }
            body {
              margin: 0;
              padding: 0;
              background: #ffffff;
              font-family: ${fontFamily}, Arial, sans-serif;
              color: ${lyricColor};
              -webkit-font-smoothing: antialiased;
            }
            .header {
              margin-bottom: 20px;
              border-bottom: 1px solid #e5e7eb;
              padding-bottom: 10px;
            }
            .title {
              font-size: ${tamanhoTitulo}px;
              font-weight: 800;
              margin: 0;
              line-height: 1.2;
              text-transform: uppercase;
              color: ${titleColor} !important;
            }
            .artist {
              font-size: ${tamanhoArtista}px;
              font-weight: 700;
              margin: 4px 0 0 0;
              line-height: 1.3;
              color: ${artistColor} !important;
            }
            .lyrics-container {
              width: 100%;
              column-count: ${columns};
              column-gap: 36px;
              font-size: ${baseFontSize}px;
              line-height: 1.5;
              color: ${lyricColor} !important;
            }
            .stanza {
              margin: 0 0 ${espacamentoEstrofe}px 0;
              white-space: pre-line;
              break-inside: avoid;
              page-break-inside: avoid;
              display: block;
            }
            .footer {
              margin-top: 24px;
              padding-top: 8px;
              font-size: 11px;
              color: #888888;
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1 class="title">${escapeHtml(title)}</h1>
            <h2 class="artist">${escapeHtml(artist)}</h2>
          </div>

          <div class="lyrics-container">
            ${stanzas.map((stanza) => `<p class="stanza">${escapeHtml(stanza.join('\n'))}</p>`).join('')}
          </div>

          ${compositor ? `<div class="footer">Composição: ${escapeHtml(compositor)}</div>` : ''}
        </body>
      </html>
    `);
    doc.close();

    setTimeout(() => {
      printIframe.contentWindow?.focus();
      printIframe.contentWindow?.print();
    }, 200);
  };

  return (
    <div className="preview-page-shell">
      <div id="lyrics-sheet" className="preview-sheet" style={{ fontFamily }}>
        <div className="preview-sheet-header">
          <h1 style={{ color: titleColor }}>{title}</h1>
          <h2 style={{ color: artistColor }}>{artist}</h2>
        </div>

        <div
          className="preview-lyrics"
          style={{
            color: lyricColor,
            fontSize: `${baseFontSize}px`,
            columnCount: columns,
            columnGap: '2.5rem',
          }}
        >
          {stanzas.map((stanza, stanzaIndex) => (
            <p key={`${stanzaIndex}-stanza`} className="preview-stanza" style={{ breakInside: 'avoid', pageBreakInside: 'avoid' }}>
              {stanza.join('\n')}
            </p>
          ))}
        </div>

        <div className="preview-sheet-footer">Composição: {artist}</div>
      </div>

      <aside className="preview-panel no-print">
        <div className="preview-panel-header">
          <span>Visualização</span>
          <button type="button" onClick={() => navigate(-1)} className="preview-back-link">
            Voltar
          </button>
        </div>

        <div className="preview-control-group">
          <label>Fonte</label>
          <select value={fontFamily} onChange={(event) => setFontFamily(event.target.value)}>
            <option value="Arial">Arial</option>
            <option value="Helvetica">Helvetica</option>
            <option value="Roboto">Roboto</option>
            <option value="Times New Roman">Times New Roman</option>
            <option value="Courier New">Courier</option>
          </select>
        </div>

        <div className="preview-control-group">
          <label>Tamanho</label>
          <div className="preview-buttongroup">
            <button type="button" onClick={() => setBaseFontSize((previous) => Math.max(10, previous - 1))}>
              A -
            </button>
            <button type="button" onClick={() => setBaseFontSize((previous) => Math.min(24, previous + 1))}>
              A +
            </button>
          </div>
        </div>

        <div className="preview-control-group">
          <label>Colunas</label>
          <div className="preview-buttongroup">
            {[1, 2].map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setColumns(option)}
                className={columns === option ? 'is-active' : ''}
              >
                {option}
              </button>
            ))}
          </div>
        </div>

        <div className="preview-color-row">
          <div className="preview-color-field">
            <span>Título</span>
            <input type="color" value={titleColor} onChange={(event) => setTitleColor(event.target.value)} />
          </div>
          <div className="preview-color-field">
            <span>Artista</span>
            <input type="color" value={artistColor} onChange={(event) => setArtistColor(event.target.value)} />
          </div>
          <div className="preview-color-field">
            <span>Letra</span>
            <input type="color" value={lyricColor} onChange={(event) => setLyricColor(event.target.value)} />
          </div>
        </div>

        <button type="button" className="preview-print-button" onClick={handleImprimir}>
          Imprimir
        </button>

        <button type="button" className="preview-reset-button" onClick={resetDefaults}>
          Resetar configurações
        </button>
      </aside>
    </div>
  );
}

export default LyricsPrintPage;
