#!/usr/bin/env python3
"""Faux binaire de resolution qui ne se termine jamais, pour tester le timeout.

Il imite juste assez VROOM pour etre lance a sa place, puis il engendre un
enfant et dort. C'est exactement le cas que le wrapper doit savoir traiter :
tuer le processus ET sa descendance, sans laisser de zombie.

Ce fichier n'est utilise que par le scenario de timeout. Il n'est jamais
appele par le service.
"""

import os
import sys
import time


def main():
    if "--version" in sys.argv or "-v" in sys.argv:
        # Repondre vite : le wrapper interroge la version avec un timeout.
        sys.stdout.write("vroom slow-stub (jamais utilise en production)\n")
        return 0

    # Un enfant qui dort : si le wrapper ne tuait que le processus direct,
    # celui-ci survivrait et le test le verrait.
    pid = os.fork()
    if pid == 0:
        while True:
            time.sleep(3600)
        return 0

    while True:
        time.sleep(3600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
