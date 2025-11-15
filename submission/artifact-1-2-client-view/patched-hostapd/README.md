# Patched-Hostapd

File ./hostap-patched.tar.xz contains the patched source for the used hostapd
in our system.

## Decompress
```bash
tar -xf hostap-patched.tar.xz
```

## Config
The config `.config` is already placed in the directory
`hostap-copy/hostap/.config`

## Compilation
Run the Makefile script
```bash
cd hostap-copy/hostap
make
```

## Hostapd Config
The hostapd runtime config is present in that same directory as `hostapd.conf`
Note: remember to change the interface name to your interface name in the config file
Example: `interface=wlan0`

## Run
```bash
sudo ./hostapd -dd hostapd.conf
```

## Run the hostapd controller
```
sudo ./hostapd_ctrl -i <your interface>
```
