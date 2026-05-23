import { defineConfig } from "vitepress";

export default defineConfig({
  title: "clifra",
  description: "A layout-first Clifford algebra framework for PyTorch",
  base: "/clifra/",
  cleanUrls: true,
  head: [["link", { rel: "icon", href: "/clifra/logo.svg" }]],
  markdown: {
    lineNumbers: true,
  },
  themeConfig: {
    logo: "/logo.svg",
    siteTitle: "clifra",
    search: {
      provider: "local",
    },
    nav: [
      { text: "Framework", link: "/framework/" },
      { text: "Guide", link: "/guide/layouts" },
      { text: "API", link: "/api/" },
      { text: "Demo", link: "/demo" },
    ],
    sidebar: [
      {
        text: "Start",
        items: [
          { text: "Overview", link: "/" },
          { text: "Framework", link: "/framework/" },
          { text: "Live Demo", link: "/demo" },
        ],
      },
      {
        text: "Guide",
        items: [
          { text: "Layouts", link: "/guide/layouts" },
          { text: "Layers", link: "/guide/layers" },
          { text: "Criteria", link: "/guide/criteria" },
        ],
      },
      {
        text: "Reference",
        items: [{ text: "API Map", link: "/api/" }],
      },
    ],
    socialLinks: [{ icon: "github", link: "https://github.com/Concode0/clifra" }],
    footer: {
      message: "Released under the Apache-2.0 License.",
      copyright: "Copyright (C) 2026 Eunkyum Kim",
    },
  },
});
