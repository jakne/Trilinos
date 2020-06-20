C    Copyright(C) 1999-2020 National Technology & Engineering Solutions
C    of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
C    NTESS, the U.S. Government retains certain rights in this software.
C    
C    See packages/seacas/LICENSE for details

C $Id: extnd5.f,v 1.2 1998/07/14 18:18:53 gdsjaar Exp $
C $Log: extnd5.f,v $
C Revision 1.2  1998/07/14 18:18:53  gdsjaar
C Removed unused variables, cleaned up a little.
C
C Changed BLUE labels to GREEN to help visibility on black background
C (indirectly requested by a couple users)
C
C Revision 1.1.1.1  1990/11/30 11:07:14  gdsjaar
C FASTQ Version 2.0X
C
c Revision 1.1  90/11/30  11:07:13  gdsjaar
c Initial revision
c
C
CC* FILE: [.PAVING]EXTND5.FOR
CC* MODIFIED BY: TED BLACKER
CC* MODIFICATION DATE: 7/6/90
CC* MODIFICATION: COMPLETED HEADER INFORMATION
C
      SUBROUTINE EXTND5 (MXND, XN, YN, ANGLE, N1, N2, N3, X, Y, DIST)
C***********************************************************************
C
C  SUBROUTINE EXTND5 = CALCULATES TWO POSITIONS AN AVERAGE LENGTH AWAY
C                      FROM A CORNER NODE AND AT 1/3 ANGLE INTERVALS
C
C***********************************************************************
C
      DIMENSION XN (MXND), YN (MXND)
      DIMENSION X(5), Y(5)
C
      ANG = ATAN2 (YN (N1) - YN (N2), XN (N1) - XN (N2))
      CANG = ANGLE * .25
      ANG1 = ANG - CANG
      ANG2 = ANG - 3. * (ANGLE * .125)
      ANG3 = ANG - 2 * CANG
      ANG4 = ANG - 5. * (ANGLE * .125)
      ANG5 = ANG - 3 * CANG
      DIST1 = SQRT ((YN (N2) - YN (N1)) **2 +  (XN (N2) - XN (N1)) **2)
      DIST2 = SQRT ((YN (N3) - YN (N2)) **2 +  (XN (N3) - XN (N2)) **2)
      DIST = (DIST1 + DIST2) * .5
      IF (CANG. EQ. 0.) THEN
         ADIST = DIST
      ELSE
         ADIST = DIST / SIN (CANG)
      ENDIF
C
      X(1) = ADIST * COS (ANG1) + XN (N2)
      Y(1) = ADIST * SIN (ANG1) + YN (N2)
      X(2) = 1.4142 * ADIST * COS (ANG2) + XN (N2)
      Y(2) = 1.4142 * ADIST * SIN (ANG2) + YN (N2)
      X(3) = ADIST * COS (ANG3) + XN (N2)
      Y(3) = ADIST * SIN (ANG3) + YN (N2)
      X(4) = 1.4142 * ADIST * COS (ANG4) + XN (N2)
      Y(4) = 1.4142 * ADIST * SIN (ANG4) + YN (N2)
      X(5) = ADIST * COS (ANG5) + XN (N2)
      Y(5) = ADIST * SIN (ANG5) + YN (N2)
C
      RETURN
C
      END
