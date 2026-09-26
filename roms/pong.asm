; Self-playing Pong. Left paddle: keys 1 (up) / 4 (down), otherwise AI.
; Right paddle: AI that only moves every other frame, so it can lose.
;
; v0-v2 scratch (BCD)   v3 left paddle y   v4 right paddle y
; v5 score L  v6 score R  v7 frame counter  v8 v9 ve temp
; va ball x  vb ball y  vc dx  vd dy

start:
    cls
    ld v3, 13
    ld v4, 13
    ld v5, 0
    ld v6, 0
    call draw_scores
    call draw_paddles

serve:
    ld va, 32
    rnd vb, 15
    add vb, 8
    ld vc, 1
    rnd v8, 1
    se v8, 0
    ld vc, 0xFF
    ld vd, 1
    rnd v8, 1
    se v8, 0
    ld vd, 0xFF
    ld i, dot
    drw va, vb, 1

loop:
    ld v8, 2
    ld dt, v8
    ld i, paddle           ; each object is erased and redrawn back-to-back
    ld ve, 1               ; so it is never invisible for a whole frame
    drw ve, v3, 6
    call move_left
    drw ve, v3, 6
    ld ve, 62
    drw ve, v4, 6
    call move_right
    drw ve, v4, 6
    ld i, dot
    drw va, vb, 1
    add va, vc
    add vb, vd
    drw va, vb, 1
    se vb, 0
    jp nb_top
    ld vd, 1
nb_top:
    se vb, 31
    jp nb_bot
    ld vd, 0xFF
nb_bot:

    se va, 2               ; left paddle hit?
    jp check_right
    se vc, 0xFF
    jp check_right
    ld v8, vb
    sub v8, v3
    se vf, 1
    jp check_right
    ld v9, 5
    sub v9, v8
    se vf, 1
    jp check_right
    ld vc, 1
    ld v8, 3
    ld st, v8

check_right:
    se va, 61
    jp check_score
    se vc, 1
    jp check_score
    ld v8, vb
    sub v8, v4
    se vf, 1
    jp check_score
    ld v9, 5
    sub v9, v8
    se vf, 1
    jp check_score
    ld vc, 0xFF
    ld v8, 3
    ld st, v8

check_score:
    se va, 0
    jp cs_right
    call draw_scores       ; erase old score
    add v6, 1
    jp scored
cs_right:
    se va, 63
    jp wait
    call draw_scores
    add v5, 1
scored:
    ld i, dot
    drw va, vb, 1          ; remove the ball that went out
    sne v5, 10
    ld v5, 0
    sne v6, 10
    ld v6, 0
    call draw_scores
    jp serve

wait:
    ld v8, dt
    se v8, 0
    jp wait
    add v7, 1
    jp loop

move_left:
    ld v8, 1
    skp v8
    jp ml_notup
    se v3, 0
    add v3, 0xFF
    ret
ml_notup:
    ld v8, 4
    skp v8
    jp ml_ai
    se v3, 26
    add v3, 1
    ret
ml_ai:
    ld v8, v3
    add v8, 2
    ld v9, vb
    sub v9, v8
    se vf, 1
    jp ml_up
    se v9, 0
    jp ml_down
    ret
ml_up:
    se v3, 0
    add v3, 0xFF
    ret
ml_down:
    se v3, 26
    add v3, 1
    ret

move_right:
    ld v8, v7
    ld v9, 1
    and v8, v9
    se v8, 1
    ret
    ld v8, v4
    add v8, 2
    ld v9, vb
    sub v9, v8
    se vf, 1
    jp mr_up
    se v9, 0
    jp mr_down
    ret
mr_up:
    se v4, 0
    add v4, 0xFF
    ret
mr_down:
    se v4, 26
    add v4, 1
    ret

draw_paddles:
    ld i, paddle
    ld ve, 1
    drw ve, v3, 6
    ld ve, 62
    drw ve, v4, 6
    ret

draw_scores:
    ld v8, 1
    ld i, bcd
    ld b, v5
    ld v2, [i]
    ld f, v2
    ld ve, 22
    drw ve, v8, 5
    ld i, bcd
    ld b, v6
    ld v2, [i]
    ld f, v2
    ld ve, 38
    drw ve, v8, 5
    ret

dot:    db 0x80
paddle: db 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
bcd:    db 0, 0, 0
