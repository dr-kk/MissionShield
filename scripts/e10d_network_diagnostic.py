#!/usr/bin/env python3
"""MissionShield E10-D network diagnostic. No scientific data are recorded."""
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import TCLink
from mininet.log import setLogLevel
import time

N=["sensor","tactical_link","edge_processing","fusion","command","actuator"]
S=["sen","tlink","edge","fus","cmd","act"]

def main():
    net=Mininet(controller=None,switch=OVSBridge,link=TCLink,autoSetMacs=True)
    h={n:net.addHost(s) for n,s in zip(N,S)}
    links=[]
    for i in range(5):
        sw=net.addSwitch(f"s{i+1}")
        l=net.addLink(h[N[i]],sw,cls=TCLink,intfName1=f"{S[i]}e{i}",
                      intfName2=f"s{i+1}a",bw=20,delay="2ms",loss=0)
        r=net.addLink(sw,h[N[i+1]],cls=TCLink,intfName1=f"s{i+1}b",
                      intfName2=f"{S[i+1]}e{i}",bw=20,delay="2ms",loss=0)
        links.append((l,r))
    net.start()
    try:
        pairs={}
        print("\n=== ADDRESS CONFIGURATION ===")
        for i,(a,b) in enumerate(zip(N[:-1],N[1:])):
            l,r=links[i]
            ia=l.intf1 if l.intf1.node==h[a] else l.intf2
            ib=r.intf1 if r.intf1.node==h[b] else r.intf2
            A=f"10.{i+1}.0.1"; B=f"10.{i+1}.0.2"
            h[a].cmd(f"ip addr flush dev {ia}; ip addr add {A}/24 dev {ia}; ip link set {ia} up")
            h[b].cmd(f"ip addr flush dev {ib}; ip addr add {B}/24 dev {ib}; ip link set {ib} up")
            pairs[(a,b)]=(A,B)
            print(a, ia, A, "<->", b, ib, B)

        print("\n=== ROUTES ===")
        for n in N:
            print(f"\n[{n}]")
            print(h[n].cmd("ip -br addr; ip route"))

        print("\n=== ADJACENCY PINGS ===")
        ok=True
        for a,b in zip(N[:-1],N[1:]):
            A,B=pairs[(a,b)]
            out=h[a].cmd(f"LC_ALL=C ping -n -c 2 -W 1 {B}")
            passed=("0% packet loss" in out)
            print(f"{a} -> {b}: {'PASS' if passed else 'FAIL'}")
            if not passed: print(out)
            ok &= passed

        print("\n=== UDP HOP-BY-HOP ===")
        port=7999
        for a,b in zip(N[:-1],N[1:]):
            A,B=pairs[(a,b)]
            outfile=f"/tmp/e10diag_{S[N.index(b)]}.txt"
            h[b].cmd(f"rm -f {outfile}; timeout 3 python3 -c "
                     +repr("import socket,sys;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
                           f"s.bind(({B!r},{port}));d,a=s.recvfrom(2048);open({outfile!r},'wb').write(d)")
                     +" >/tmp/e10diag_rx.log 2>&1 &")
            time.sleep(.15)
            h[a].cmd("python3 -c "+repr(
                "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
                f"s.bind(({A!r},0));s.sendto(b'E10DIAG',({B!r},{port}))"))
            time.sleep(.2)
            got=h[b].cmd(f"cat {outfile} 2>/dev/null || true").strip()
            passed=(got=="E10DIAG")
            print(f"{a} -> {b}: {'PASS' if passed else 'FAIL'} received={got!r}")
            if not passed:
                print("RX log:",h[b].cmd("cat /tmp/e10diag_rx.log 2>/dev/null || true"))
            ok &= passed

        print("\nDIAGNOSTIC RESULT:", "PASS" if ok else "FAIL")
        print("No MissionShield inference and no E10-D scientific outcome recorded.")
    finally:
        net.stop()

if __name__=="__main__":
    setLogLevel("warning")
    main()
