import { Config } from "fumadocs-python-autodoc/source";

const config: Config = {
  shiki: {
    lang: "python",
    themes: {
      dark: "vitesse-dark",
      light: "vitesse-light",
    },
  },
  jsonPath: "lib",
  sources: {
    tatva: {
      baseUrl: "api",
      title: "API Reference",
      pkgName: "tatva",
      options: {
        className: "route-api",
      },
      sortClassMethods: true,
      gitUrl: "https://https://github.com/smec-ethz/tatva/tree/main/tatva",
      excludeModules: [],
    },
  },
};

export default config;
