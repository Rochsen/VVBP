import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      'dayjs/plugin/advancedFormat': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/advancedFormat/index.js'),
      'dayjs/plugin/advancedFormat.js': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/advancedFormat/index.js'),
      'dayjs/plugin/customParseFormat': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/customParseFormat/index.js'),
      'dayjs/plugin/customParseFormat.js': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/customParseFormat/index.js'),
      'dayjs/plugin/localeData': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/localeData/index.js'),
      'dayjs/plugin/localeData.js': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/localeData/index.js'),
      'dayjs/plugin/weekday': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/weekday/index.js'),
      'dayjs/plugin/weekday.js': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/weekday/index.js'),
      'dayjs/plugin/weekOfYear': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/weekOfYear/index.js'),
      'dayjs/plugin/weekOfYear.js': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/weekOfYear/index.js'),
      'dayjs/plugin/weekYear': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/weekYear/index.js'),
      'dayjs/plugin/weekYear.js': path.resolve(__dirname, 'node_modules/.pnpm/dayjs@1.11.20/node_modules/dayjs/esm/plugin/weekYear/index.js'),
    },
  },
});
