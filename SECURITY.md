# Security policy

## Supported versions

| Version | Supported |
|---|---|
| `v1.3.0` | yes |
| `v1.2.0` | yes, research baseline |

## Reporting a vulnerability

Report privately to **dhruvjkotian@gmail.com**. Include the version, the board
configuration, a reproduction and the impact. Do not open a public issue for a
memory-safety or parsing flaw.

## Scope

The firmware is offline by design: no Wi-Fi, no cloud, no network service. The
relevant attack surface is local:

- **Model parsing.** The `.m8` and `MOE1` readers trust header lengths, offsets,
  CRCs and SHA-256 values. A malformed blob on an SD card is the main untrusted
  input. Do not load blobs from untrusted sources.
- **SD card.** The card is mounted only while staging a model and unmounted
  before inference. A malicious filesystem is still parsed by FATFS.
- **Serial console.** The UART console is a debug interface, not an
  authentication boundary.

## What this project does not provide

- Secure boot, flash encryption or signed models. SHA-256 is an integrity check,
  not authentication.
- Any guarantee that a 260K model is safe for actuation. Do not put this model
  between a sensor and an actuator without a deterministic safety layer.

## Repository hygiene

No credentials, Wi-Fi details, MAC addresses, private IPs, home paths or flash
dumps are tracked. If you find one, report it as above.
