{pkgs}: {
  deps = [
    pkgs.which
    pkgs.libpng
    pkgs.libjpeg_turbo
    pkgs.postgresql
    pkgs.openssl
  ];
}
