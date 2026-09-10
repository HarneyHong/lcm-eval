

DROP TABLE IF EXISTS "Bio";

CREATE TABLE "Bio" (
  "fibros" integer ,
  "activity" integer ,
  "b_id" integer ,
  PRIMARY KEY ("b_id")
) ;

DROP TABLE IF EXISTS "dispat";

CREATE TABLE "dispat" (
  "m_id" integer  DEFAULT 0,
  "sex" integer DEFAULT NULL,
  "age" integer DEFAULT NULL,
  "Type" integer DEFAULT NULL,
  PRIMARY KEY ("m_id")
) ;

DROP TABLE IF EXISTS "indis";

CREATE TABLE "indis" (
  "got" integer DEFAULT NULL,
  "gpt" integer DEFAULT NULL,
  "alb" integer DEFAULT NULL,
  "tbil" integer DEFAULT NULL,
  "dbil" integer DEFAULT NULL,
  "che" integer DEFAULT NULL,
  "ttt" integer DEFAULT NULL,
  "ztt" integer DEFAULT NULL,
  "tcho" integer DEFAULT NULL,
  "tp" integer DEFAULT NULL,
  "in_id" integer ,
  PRIMARY KEY ("in_id")
) ;

DROP TABLE IF EXISTS "inf";

CREATE TABLE "inf" (
  "dur" integer DEFAULT NULL,
  "a_id" integer  DEFAULT 0,
  PRIMARY KEY ("a_id")
) ;

DROP TABLE IF EXISTS "rel11";

CREATE TABLE "rel11" (
  "b_id" integer  DEFAULT 0,
  "m_id" integer  DEFAULT 0,
  PRIMARY KEY ("b_id","m_id")
) ;

DROP TABLE IF EXISTS "rel12";

CREATE TABLE "rel12" (
  "in_id" integer  DEFAULT 0,
  "m_id" integer  DEFAULT 0,
  PRIMARY KEY ("in_id","m_id")
) ;

DROP TABLE IF EXISTS "rel13";

CREATE TABLE "rel13" (
  "a_id" integer  DEFAULT 0,
  "m_id" integer  DEFAULT 0,
  PRIMARY KEY ("a_id","m_id")
) ;
