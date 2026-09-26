; Held-out demo: a bouncing smiley + a frame counter (0-255, shown as 3 digits).
; Uses instructions Pong barely touches: SHR/SHL, OR, STORE/LOAD, ADD I.
;
; v0-v2 digits scratch   v3 x  v4 y  v5 dx  v6 dy  v7 frame counter
; v8 v9 temp             va,vb saved copy of x,y (via STORE/LOAD)

start:
    cls
    ld v3, 10
    ld v4, 6
    ld v5, 1
    ld v6, 1
    ld v7, 0
    call digits
    ld i, smiley
    drw v3, v4, 6

loop:
    ld v8, 1
    ld dt, v8
    ld i, smiley
    drw v3, v4, 6        ; erase
    add v3, v5
    add v4, v6
    drw v3, v4, 6        ; draw at new spot
    se v3, 0
    jp nx0
    ld v5, 1
nx0:
    se v3, 56
    jp nx1
    ld v5, 0xFF
nx1:
    se v4, 8
    jp ny0
    ld v6, 1
ny0:
    se v4, 26
    jp ny1
    ld v6, 0xFF
ny1:
    call digits          ; erase old counter
    add v7, 1
    call digits          ; draw new counter
    ld i, save           ; round-trip x,y through memory for no reason but testing
    ld v8, v3
    ld v9, v4
    ld va, v8
    ld vb, v9
    shl va
    shr va
    ld v8, 0
    or v8, vb
    ld i, save
    add i, v8            ; I = save + y  (never used; just exercises ADD I)
wait:
    ld v8, dt
    se v8, 0
    jp wait
    jp loop

digits:
    ld i, bcd
    ld b, v7
    ld v2, [i]
    ld v8, 50
    ld v9, 1
    ld f, v0
    drw v8, v9, 5
    add v8, 5
    ld f, v1
    drw v8, v9, 5
    add v8, 5
    ld f, v2
    drw v8, v9, 5
    ld i, bcd
    ld [i], v2           ; STORE the digits back (same values)
    ret

smiley: db 0x3C, 0x42, 0xA5, 0x81, 0xA5, 0x5A
bcd:    db 0, 0, 0
save:   db 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
