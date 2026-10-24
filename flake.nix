{
  description = "ZINC — a fast LLM inference engine for Vulkan (RDNA3/RDNA4) and Metal (Apple Silicon)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        zigVersion = pkgs.lib.removeSuffix "\n" (builtins.readFile ./.zig-version);
        zigRelease = builtins.fromJSON (builtins.readFile ./nix/zig-release.json);
        zigDist = zigRelease.platforms.${system};
        # Fetch the same official release as scripts/zig.sh. The locked nixpkgs
        # predates this compiler; selecting its default Zig would silently drift.
        zigPkg = assert zigRelease.version == zigVersion; pkgs.stdenv.mkDerivation {
          pname = "zig";
          version = zigVersion;
          src = pkgs.fetchurl { inherit (zigDist) url hash; };
          dontConfigure = true;
          dontBuild = true;
          dontFixup = true;
          installPhase = ''
            runHook preInstall
            mkdir -p "$out/bin" "$out/lib"
            cp zig "$out/bin/zig"
            cp -r lib/. "$out/lib/"
            substituteInPlace "$out/lib/std/zig/system.zig" \
              --replace-fail /usr/bin/env ${pkgs.coreutils}/bin/env
            runHook postInstall
          '';
          meta = {
            description = "Zig compiler pinned by ZINC";
            homepage = "https://ziglang.org/";
            license = pkgs.lib.licenses.mit;
            platforms = builtins.attrNames zigRelease.platforms;
            mainProgram = "zig";
          };
        };

        zinc = pkgs.stdenv.mkDerivation {
          pname = "zinc";
          version = "0.1.0";

          src = ./.;

          nativeBuildInputs = [
            zigPkg
            pkgs.pkg-config
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.shaderc   # provides glslc for SPIR-V shader compilation
            pkgs.autoPatchelfHook
          ];

          buildInputs = pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.vulkan-loader
            pkgs.vulkan-headers
          ];

          # Keep compiler caches writable without changing the builder's HOME.
          preBuild = ''
            export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zinc-zig-global-cache"
            export ZIG_LOCAL_CACHE_DIR="$TMPDIR/zinc-zig-local-cache"
            mkdir -p "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR"
          '';

          buildPhase = ''
            runHook preBuild
            zig build -Doptimize=ReleaseFast --prefix $out
            runHook postBuild
          '';

          # zig build --prefix already places everything under $out,
          # so there is nothing more to install.
          dontInstall = true;

          meta = with pkgs.lib; {
            description = "ZINC LLM inference engine (Vulkan / Metal)";
            longDescription = ''
              ZINC is a high-performance LLM inference engine targeting AMD
              RDNA3/RDNA4 GPUs via Vulkan and Apple Silicon via Metal.  It
              speaks the OpenAI-compatible HTTP API and supports GGUF models
              (Qwen3, Gemma 4 MoE, and more).
            '';
            homepage = "https://github.com/zolotukhin/zinc";
            license = licenses.mit;
            maintainers = [];
            platforms = platforms.linux ++ platforms.darwin;
          };
        };
      in
      {
        packages = {
          default = zinc;
          inherit zinc;
          zig = zigPkg;
        };

        # `nix develop` shell — matches the project's manual build instructions.
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            zigPkg
            pkgs.pkg-config
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.shaderc
            pkgs.vulkan-loader
            pkgs.vulkan-headers
            pkgs.vulkan-validation-layers
          ];

          shellHook = ''
            echo "zinc dev shell — zig $(zig version)"
          '';
        };
      }
    );
}
