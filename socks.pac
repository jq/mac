function FindProxyForURL(url, host)
{
  if (host.match('^hadoop*')) {
    return "SOCKS5 127.0.0.1:8001";
  }
  return "DIRECT";
}