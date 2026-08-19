import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { jsPDF } from 'jspdf';

const MODELS = ['tiny', 'base', 'small', 'medium', 'turbo', 'large-v3'];
const normalizeYoutubeUrl = (value) => value.trim();
const getYoutubeVideoId = (value) => {
  const normalized = normalizeYoutubeUrl(value);
  if (!normalized) return '';

  const directMatch = normalized.match(/[?&]v=([^&]+)/i) || normalized.match(/youtu\.be\/([^?&]+)/i);
  if (directMatch) return directMatch[1];

  const embedMatch = normalized.match(/embed\/([^?&]+)/i);
  if (embedMatch) return embedMatch[1];

  const shortsMatch = normalized.match(/shorts\/([^?&]+)/i);
  if (shortsMatch) return shortsMatch[1];

  return '';
};
const PDF_FONT_OPTIONS = [
  { label: 'Helvetica', value: 'helvetica' },
  { label: 'Times', value: 'times' },
  { label: 'Courier', value: 'courier' },
];
const PAGE_FORMAT_OPTIONS = [
  { label: 'A4', value: 'a4' },
  { label: 'Carta', value: 'letter' },
  { label: 'A5', value: 'a5' },
];
const COLUMN_OPTIONS = [
  { label: '1 coluna', value: 1 },
  { label: '2 colunas', value: 2 },
  { label: '3 colunas', value: 3 },
];

function App() {
  const navigate = useNavigate();
  const [title, setTitle] = useState('Título da Música');
  const [artist, setArtist] = useState('Cantor Desconhecido');
  const [model, setModel] = useState('small');
  const [pauseSeconds, setPauseSeconds] = useState(2.5);
  const [pdfFont, setPdfFont] = useState('helvetica');
  const [titleFontSize, setTitleFontSize] = useState(22);
  const [artistFontSize, setArtistFontSize] = useState(12);
  const [lyricFontSize, setLyricFontSize] = useState(12);
  const [pageFormat, setPageFormat] = useState('a4');
  const [pageOrientation, setPageOrientation] = useState('portrait');
  const [pageMargin, setPageMargin] = useState(48);
  const [lineSpacing, setLineSpacing] = useState(6);
  const [textAlign, setTextAlign] = useState('center');
  const [columnCount, setColumnCount] = useState(2);
  const [inputMode, setInputMode] = useState('youtube');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isPreparingAudio, setIsPreparingAudio] = useState(false);
  const [youtubeMeta, setYoutubeMeta] = useState({ title: '', author: '', thumbnail: '' });
  const [error, setError] = useState('');
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const hasTranscript = useMemo(() => transcript.trim().length > 0, [transcript]);
  const audioPreviewUrl = useMemo(
    () => (audioFile ? URL.createObjectURL(audioFile) : ''),
    [audioFile],
  );
  const videoId = useMemo(() => getYoutubeVideoId(youtubeUrl), [youtubeUrl]);
  const youtubeThumbnail = useMemo(() => {
    if (youtubeMeta.thumbnail) return youtubeMeta.thumbnail;
    return videoId ? `https://img.youtube.com/vi/${videoId}/hqdefault.jpg` : '';
  }, [videoId, youtubeMeta.thumbnail]);
  const youtubeEmbedUrl = useMemo(
    () => (videoId ? `https://www.youtube.com/embed/${videoId}?rel=0` : ''),
    [videoId],
  );

  useEffect(() => {
    const value = normalizeYoutubeUrl(youtubeUrl);
    if (!value) {
      setYoutubeMeta({ title: '', author: '', thumbnail: '' });
      return undefined;
    }

    let isCancelled = false;

    const fetchMeta = async () => {
      try {
        const response = await fetch(`https://noembed.com/embed?url=${encodeURIComponent(value)}`);
        if (!response.ok) return;
        const payload = await response.json();

        if (isCancelled) return;

        const nextTitle = payload.title || '';
        const nextAuthor = payload.author_name || payload.author || '';
        const nextThumbnail = payload.thumbnail_url || (videoId ? `https://img.youtube.com/vi/${videoId}/hqdefault.jpg` : '');

        setYoutubeMeta({
          title: nextTitle,
          author: nextAuthor,
          thumbnail: nextThumbnail,
        });
      } catch (fetchError) {
        if (!isCancelled && videoId) {
          setYoutubeMeta({
            title: '',
            author: '',
            thumbnail: `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`,
          });
        }
      }
    };

    fetchMeta();

    return () => {
      isCancelled = true;
    };
  }, [videoId, youtubeUrl]);

  const handleTranscribe = async () => {
    if (isLoading) return;
    setError('');

    const trimmedUrl = youtubeUrl.trim();
    if (inputMode === 'youtube' && !trimmedUrl) {
      setError('Cole um link do YouTube antes de continuar.');
      return;
    }

    if (inputMode === 'file' && !audioFile) {
      setError('Selecione um arquivo de áudio antes de transcrever.');
      return;
    }

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('model', model);
      formData.append('pause', String(pauseSeconds));

      if (inputMode === 'youtube') {
        formData.append('youtubeUrl', trimmedUrl);
      } else {
        formData.append('audio', audioFile);
      }

      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || 'Não foi possível concluir a transcrição.');
      }

      setTranscript(payload.transcript || '');

      if (payload.title) {
        setTitle(payload.title);
      }

      if (payload.artist) {
        setArtist(payload.artist);
      }
    } catch (fetchError) {
      setError(fetchError.message || 'Erro ao processar a transcrição.');
    } finally {
      setIsLoading(false);
    }
  };

  const getPreviewLines = () => {
    const parsed = transcript
      .replace(/\r/g, '')
      .split('\n')
      .map((line) => line.trimEnd());

    return parsed.length ? parsed : [''];
  };

  const getLyricStanzas = () => {
    const normalized = transcript.replace(/\r/g, '').trim();
    if (!normalized) return [['']];

    return normalized
      .split(/\n\s*\n+/)
      .map((stanza) => stanza.split('\n').map((line) => line.trim()).filter(Boolean));
  };

  const getPreviewPages = () => {
    const lines = getPreviewLines();
    const pageSize = 16;
    const chunks = [];

    for (let index = 0; index < lines.length; index += pageSize) {
      chunks.push(lines.slice(index, index + pageSize));
    }

    return chunks.length ? chunks : [['']];
  };

  const generatePdf = (options = {}) => {
    const fontName = options.pdfFont || pdfFont;
    const titleSize = options.titleFontSize ?? titleFontSize;
    const artistSize = options.artistFontSize ?? artistFontSize;
    const linesSize = options.lyricFontSize ?? lyricFontSize;
    const orientation = options.pageOrientation || pageOrientation;
    const format = options.pageFormat || pageFormat;
    const margin = options.pageMargin ?? pageMargin;
    const spacing = options.lineSpacing ?? lineSpacing;
    const align = options.textAlign || textAlign;

    const pdf = new jsPDF({ unit: 'pt', format, orientation });
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const contentWidth = pageWidth - margin * 2;
    const titleX = align === 'left' ? margin : pageWidth / 2;

    pdf.setFillColor(255, 255, 255);
    pdf.rect(0, 0, pageWidth, pageHeight, 'F');
    pdf.setDrawColor(210, 214, 220);
    pdf.setLineWidth(0.7);
    pdf.line(margin, 50, pageWidth - margin, 50);

    pdf.setTextColor(17, 24, 39);
    pdf.setFont(fontName, 'bold');
    pdf.setFontSize(titleSize);
    pdf.text((title || 'Título da Música').toUpperCase(), titleX, 38, { align });

    pdf.setFont(fontName, 'normal');
    pdf.setFontSize(artistSize);
    pdf.setTextColor(71, 85, 105);
    pdf.text(artist || 'Cantor Desconhecido', titleX, 66, { align });

    pdf.setDrawColor(226, 232, 240);
    pdf.setLineWidth(0.5);
    pdf.line(margin, 78, pageWidth - margin, 78);

    pdf.setTextColor(15, 23, 42);
    pdf.setFont(fontName, 'normal');
    pdf.setFontSize(linesSize);

    let y = 98;
    const safeWidth = Math.max(contentWidth - 18, 90);
    const lines = pdf.splitTextToSize(transcript, safeWidth);

    lines.forEach((line) => {
      if (y > pageHeight - 32) {
        pdf.addPage();
        y = 44;
      }

      pdf.text(line, align === 'left' ? margin : pageWidth / 2, y, { align });
      y += linesSize + spacing;
    });

    return pdf;
  };

  const handleDownloadPdf = () => {
    if (!transcript.trim()) {
      setError('Primeiro gere uma transcrição para exportar o PDF.');
      return;
    }

    generatePdf().save(`${(title || 'letra').trim() || 'letra'}.pdf`);
  };

  const handleClear = () => {
    setTranscript('');
    setAudioFile(null);
    setYoutubeUrl('');
    setError('');
    setTitle('Título da Música');
    setArtist('Cantor Desconhecido');
  };

  return (
    <div className="app-shell">
      <div className="mobile-tools-bar">
        <button
          type="button"
          className="mobile-toggle-button"
          onClick={() => setMobileSidebarOpen((previous) => !previous)}
          aria-expanded={mobileSidebarOpen}
        >
          {mobileSidebarOpen ? 'Ocultar opções' : 'Mostrar opções'}
        </button>
      </div>

      <aside className={mobileSidebarOpen ? 'sidebar mobile-open' : 'sidebar'}>
        <div className="sidebar-header">
          <p className="eyebrow">Tudo em 3 passos</p>
          <h2>⚙️ Opções</h2>
        </div>

        <div className="list-steps">
          <div className="step-item"><span>1</span>Defina o nome</div>
          <div className="step-item"><span>2</span>Escolha a fonte</div>
          <div className="step-item"><span>3</span>Baixe a letra</div>
        </div>

        <label className="field">
          <span>Título da Música</span>
          <input value={title} onChange={(event) => setTitle(event.target.value)} />
        </label>

        <label className="field">
          <span>Artista</span>
          <input value={artist} onChange={(event) => setArtist(event.target.value)} />
        </label>

        <label className="field">
          <span>Precisão (Modelo)</span>
          <select value={model} onChange={(event) => setModel(event.target.value)}>
            {MODELS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label className="field range-field">
          <span>Sensibilidade de Estrofe</span>
          <div className="range-row">
            <input
              type="range"
              min="1"
              max="5"
              step="0.1"
              value={pauseSeconds}
              onChange={(event) => setPauseSeconds(Number(event.target.value))}
            />
            <strong>{pauseSeconds.toFixed(1)}s</strong>
          </div>
        </label>

        <div className="info-box">
          O modelo <strong>small</strong> é o melhor custo-benefício entre velocidade e precisão.
        </div>
        <div className="info-box">
          Ajuste a sensibilidade de estrofe para controlar quebras na letra transcrita.
        </div>
      </aside>

      <main className="content">
        <header className="page-header">
          <h1>Transcreva a letra de músicas em poucos cliques</h1>
          <p>
            Cole o link do YouTube ou envie um arquivo local e transforme o áudio em letra pronta para PDF.
          </p>
        </header>

        <div className="panel-grid">
          <section className="panel source-panel">
            <div className="panel-header-row">
              <h3>🎧 Fonte do áudio</h3>
            </div>

            <div className="segmented-control" role="tablist" aria-label="Fonte do áudio">
              <button
                type="button"
                className={inputMode === 'youtube' ? 'segment active' : 'segment'}
                onClick={() => setInputMode('youtube')}
              >
                YouTube
              </button>
              <button
                type="button"
                className={inputMode === 'file' ? 'segment active' : 'segment'}
                onClick={() => setInputMode('file')}
              >
                Arquivo local
              </button>
            </div>

            {inputMode === 'youtube' ? (
              <label className="field url-field">
                <span>Link do YouTube</span>
                <input
                  type="url"
                  value={youtubeUrl}
                  onChange={(event) => setYoutubeUrl(event.target.value)}
                  placeholder="https://youtu.be/..."
                />
              </label>
            ) : (
              <>
                <label className="upload-box">
                  <input
                    type="file"
                    accept=".mp3,.wav,.m4a,audio/mpeg,audio/wav,audio/mp4"
                    onChange={(event) => {
                      setIsPreparingAudio(true);
                      setAudioFile(event.target.files?.[0] || null);
                      window.setTimeout(() => setIsPreparingAudio(false), 300);
                    }}
                  />
                  <span>
                    {audioFile
                      ? `Arquivo selecionado: ${audioFile.name}`
                      : isPreparingAudio
                        ? 'Preparando áudio...'
                        : 'Arraste seu áudio (MP3, WAV, M4A)'}
                  </span>
                </label>

                {audioFile && (
                  <div className="audio-preview-card">
                    <div className="audio-preview-status">
                      {isPreparingAudio ? 'Preparando áudio...' : 'Áudio pronto'}
                    </div>
                    <audio controls className="audio-player" src={audioPreviewUrl} />
                  </div>
                )}
              </>
            )}

            {inputMode === 'youtube' && youtubeUrl && (
              <div className="video-preview-card">
                {youtubeEmbedUrl ? (
                  <div className="video-embed-wrapper">
                    <iframe
                      src={youtubeEmbedUrl}
                      title={youtubeMeta.title || 'Preview do vídeo do YouTube'}
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                      className="youtube-embed"
                    />
                  </div>
                ) : null}

                

                <div className="video-preview-meta">
                  {youtubeMeta.title ? <span className="video-main-title">{youtubeMeta.title}</span> : null}
                  {youtubeMeta.author ? <span className="video-main-author">{youtubeMeta.author}</span> : null}
                  
                </div>
              </div>
            )}

            <button className="primary-button" onClick={handleTranscribe} disabled={isLoading || isPreparingAudio}>
              {isLoading ? 'Baixando e transcrevendo...' : isPreparingAudio ? 'Preparando áudio...' : '🚀 Transcrever letra'}
            </button>
          </section>

          <section className="panel result-panel">
            <h3>📄 Letra e Impressão</h3>

            {hasTranscript ? (
              <>
                <textarea
                  value={transcript}
                  onChange={(event) => setTranscript(event.target.value)}
                  rows={16}
                  placeholder="A letra aparecerá aqui após a transcrição..."
                />

                <div className="action-row">
                  <button
                    className="secondary-button"
                    onClick={() => {
                      if (!transcript.trim()) {
                        setError('Primeiro gere uma transcrição para abrir a visualização de impressão.');
                        return;
                      }

                      navigate('/preview', {
                        state: {
                          title,
                          artist,
                          transcript,
                        },
                      });
                    }}
                  >
                    🖨️ Imprimir / PDF
                  </button>
                  <button className="ghost-button" onClick={handleClear}>
                    🗑️ Limpar
                  </button>
                </div>
              </>
            ) : (
              <div className="empty-state">Sua letra aparecerá aqui após a transcrição.</div>
            )}

            {error && <div className="error-message">{error}</div>}
          </section>
        </div>
      </main>

    </div>
  );
}

export default App;
