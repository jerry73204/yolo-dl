#!/usr/bin/env python3
import sys

def main():
    recip = list(map(lambda arg: 1.0 / float(arg), sys.argv[1:]))

    if not recip:
        return
    
    min_val = min(recip)
    print(', '.join(map(lambda val: str(val / min_val), recip)))

if __name__ == '__main__':
    main()
