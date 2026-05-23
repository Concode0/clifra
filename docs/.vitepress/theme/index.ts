import DefaultTheme from "vitepress/theme";

import ManifoldDemo from "./components/ManifoldDemo.vue";
import "./style.css";

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component("ManifoldDemo", ManifoldDemo);
  },
};
