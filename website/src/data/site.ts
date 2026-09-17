import { withBase } from '../utils/path';

export const site = {
  brand: 'HERTZ',
  repoUrl: 'https://github.com/seetapsych/seetapsych-hertz',
  maeBpm: '3.88',
  dataset: 'VIPL-HR V1',
  subjects: 22,
  videos: 485,
  reportPage: 7,
  inputFrames: 160,
  inputSize: 128,
  reportUrl: withBase('/downloads/tinyhr-technical-report.pdf'),
  flowchartPdfUrl: withBase('/downloads/tinyhr-flowchart.pdf'),
  flowchartImageUrl: withBase('/media/tinyhr-flowchart.png'),
  heroPreviewUrl: withBase('/media/hero-preview.mp4'),
  demoUrl: withBase('/media/demo-full.mp4'),
  posterUrl: withBase('/media/demo-poster.jpg'),
  adachromPipelineUrl: withBase('/media/adachrom-pipeline.png'),
  southeastUniversityLogoUrl: withBase('/media/affiliations/southeast-university.png'),
  ictCasLogoUrl: withBase('/media/affiliations/ict-cas.png'),
} as const;
