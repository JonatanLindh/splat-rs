{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShell.${system} = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          pkg-config
          mold
        ];

        buildInputs = with pkgs; [
          wayland
          rerun
        ];

        LIBCLANG_PATH = pkgs.lib.makeLibraryPath [ pkgs.libclang ];

        LD_LIBRARY_PATH =
          with pkgs;
          lib.makeLibraryPath [
            vulkan-loader
            libxkbcommon
            wayland
          ];
      };
    };
}
